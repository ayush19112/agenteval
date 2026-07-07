"""Single entry point for all LLM calls.

Routing:
- "mock/*"  -> built-in deterministic fake (no network, no keys) used for
  tests and the zero-key demo. "mock/good" plays the tool-calling game
  properly and abstains when it has no tools; "mock/weak" skips tools and
  answers with confident fabrications.
- anything else -> LiteLLM (lazy import; install with `pip install agenteval[llm]`),
  behind a per-provider rate limiter (see core.ratelimit), with exponential
  backoff + jitter on rate limits / transient 5xx.

If retries are exhausted on a transient error and FALLBACK_MODEL is set (e.g.
a local Ollama model), the call is retried once via the fallback provider
instead of crashing the suite.
"""

import json
import os
import time

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from agenteval.core.ratelimit import get_limiter

# List prices per 1M tokens (input, output) used only when litellm cannot
# price a model itself. Local/mock inference is free.
FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "gemini/gemini-2.5-flash": (0.30, 2.50),
}


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    model: str
    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model.startswith(("mock/", "ollama/")):
        return 0.0
    if model in FALLBACK_PRICING:
        price_in, price_out = FALLBACK_PRICING[model]
        return tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
    return 0.0


async def _dispatch(
    model: str, messages: list[dict], tools: list[dict] | None, temperature: float
) -> LLMResponse:
    if model.startswith("mock/"):
        return _mock_complete(model, messages, tools or [])
    async with get_limiter(model):
        return await _litellm_complete(model, messages, tools, temperature)


async def complete(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
) -> LLMResponse:
    start = time.perf_counter()
    try:
        response = await _dispatch(model, messages, tools, temperature)
    except Exception as e:
        fallback = os.getenv("FALLBACK_MODEL")
        if not fallback or fallback == model or not _is_transient(e):
            raise
        response = await _dispatch(fallback, messages, tools, temperature)
    response.latency_ms = (time.perf_counter() - start) * 1000
    return response


# --------------------------------------------------------------------------
# Real providers via LiteLLM
# --------------------------------------------------------------------------

_TRANSIENT_ERRORS = ("RateLimit", "ServiceUnavailable", "InternalServer", "Timeout", "Connection")


def _is_transient(exc: BaseException) -> bool:
    return any(marker in type(exc).__name__ for marker in _TRANSIENT_ERRORS)


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _litellm_complete(
    model: str, messages: list[dict], tools: list[dict] | None, temperature: float
) -> LLMResponse:
    try:
        import litellm
    except ImportError as e:
        raise RuntimeError(
            f"Model '{model}' needs LiteLLM. Install it with: pip install agenteval[llm] "
            "(or use the offline models mock/good and mock/weak)."
        ) from e

    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    raw = await litellm.acompletion(**kwargs)

    choice = raw.choices[0].message
    tool_calls = [
        ToolCallRequest(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        )
        for tc in (choice.tool_calls or [])
    ]
    tokens_in = getattr(raw.usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(raw.usage, "completion_tokens", 0) or 0
    try:
        cost = litellm.completion_cost(completion_response=raw)
    except Exception:
        cost = estimate_cost(model, tokens_in, tokens_out)
    return LLMResponse(
        model=model,
        content=choice.content,
        tool_calls=tool_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
    )


# --------------------------------------------------------------------------
# Deterministic mock provider (no network, no keys)
# --------------------------------------------------------------------------

ABSTAIN_TEXT = (
    "I don't know — I don't have reliable information about this, so I can't give an answer "
    "and won't guess."
)
FABRICATED_TEXT = (
    "I'm certain about this: the weather is 25°C and sunny, the total comes to exactly "
    "12,345, and the person you asked about was born in 1970 in Springfield."
)


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mock_complete(model: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
    variant = model.split("/", 1)[1]
    tokens_in = _rough_tokens(json.dumps(messages))
    tool_results = [m.get("content", "") for m in messages if m.get("role") == "tool"]

    if variant == "weak":
        # Never uses tools; answers with confident fabrications.
        content = FABRICATED_TEXT
    elif tools and not tool_results:
        # "good": first turn with tools available -> call every tool once,
        # using the example args the benchmark author attached to each mock tool.
        calls = [
            ToolCallRequest(
                id=f"call_{i}",
                name=t["function"]["name"],
                arguments=t["function"].get("x_example_args") or {},
            )
            for i, t in enumerate(tools)
        ]
        return LLMResponse(model=model, tool_calls=calls, tokens_in=tokens_in, tokens_out=8)
    elif tool_results:
        # "good": tools have answered -> compose a final answer strictly from evidence.
        content = "; ".join(tool_results) + "."
    else:
        # "good" with no tools: refuse to guess (correct behavior on unanswerable tasks).
        content = ABSTAIN_TEXT

    return LLMResponse(
        model=model, content=content, tokens_in=tokens_in, tokens_out=_rough_tokens(content)
    )
