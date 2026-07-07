"""Phase 7: rate limiting, transient-error detection, and provider fallback."""

import asyncio
import time

import pytest

import agenteval.core.llm as llm
from agenteval.core.llm import LLMResponse, _is_transient, complete
from agenteval.core.ratelimit import RateLimiter, _parse_env_map


class FakeRateLimitError(Exception):
    pass


class FatalError(Exception):
    pass


def test_transient_detection():
    class RateLimitError(Exception): ...

    class InternalServerError(Exception): ...

    class AuthenticationError(Exception): ...

    assert _is_transient(RateLimitError())
    assert _is_transient(InternalServerError())
    assert _is_transient(TimeoutError())
    assert not _is_transient(AuthenticationError())
    assert not _is_transient(ValueError())


def test_parse_env_map(monkeypatch):
    monkeypatch.setenv("PROVIDER_RPM", "groq=28, gemini=14,broken,alsobad=x")
    assert _parse_env_map("PROVIDER_RPM") == {"groq": 28, "gemini": 14}


async def test_rate_limiter_enforces_window():
    limiter = RateLimiter(rpm=2, window=0.3)

    async def one():
        async with limiter:
            return time.monotonic()

    stamps = sorted(await asyncio.gather(one(), one(), one(), one()))
    # 2 immediately, the next 2 only after the window slides
    assert stamps[1] - stamps[0] < 0.15
    assert stamps[2] - stamps[0] >= 0.25
    assert stamps[3] - stamps[0] >= 0.25


async def test_rate_limiter_bounds_concurrency():
    limiter = RateLimiter(concurrency=2)
    active, peak = 0, 0

    async def one():
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(one() for _ in range(6)))
    assert peak == 2


async def test_fallback_on_transient_error(monkeypatch):
    calls = []

    async def fake_litellm(model, messages, tools, temperature):
        calls.append(model)
        if model == "groq/primary":
            raise FakeRateLimitError("429")
        return LLMResponse(model=model, content="from fallback")

    monkeypatch.setattr(llm, "_litellm_complete", fake_litellm)
    monkeypatch.setenv("FALLBACK_MODEL", "ollama/backup")

    response = await complete("groq/primary", [{"role": "user", "content": "hi"}])
    assert response.content == "from fallback"
    assert response.model == "ollama/backup"
    assert calls == ["groq/primary", "ollama/backup"]


async def test_no_fallback_on_fatal_error(monkeypatch):
    async def fake_litellm(model, messages, tools, temperature):
        raise FatalError("bad api key")

    monkeypatch.setattr(llm, "_litellm_complete", fake_litellm)
    monkeypatch.setenv("FALLBACK_MODEL", "ollama/backup")

    with pytest.raises(FatalError):
        await complete("groq/primary", [{"role": "user", "content": "hi"}])


async def test_no_fallback_when_unset(monkeypatch):
    async def fake_litellm(model, messages, tools, temperature):
        raise FakeRateLimitError("429")

    monkeypatch.setattr(llm, "_litellm_complete", fake_litellm)
    monkeypatch.delenv("FALLBACK_MODEL", raising=False)

    with pytest.raises(FakeRateLimitError):
        await complete("groq/primary", [{"role": "user", "content": "hi"}])
