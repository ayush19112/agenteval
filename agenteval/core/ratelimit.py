"""Per-provider rate limiting for LLM calls.

Free tiers throttle aggressively and differently per provider, so limits are
keyed by the provider prefix of the model string ("groq/..." -> "groq") and
configured via env vars:

    PROVIDER_RPM="groq=28,gemini=14"          # max requests per rolling minute
    PROVIDER_CONCURRENCY="groq=2,gemini=4"    # max in-flight requests

Limiters are held per event loop (asyncio primitives are loop-bound).
"""

import asyncio
import os
import time
import weakref
from collections import deque


def _parse_env_map(name: str) -> dict[str, int]:
    raw = os.getenv(name, "")
    out = {}
    for pair in raw.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            try:
                out[key.strip()] = int(value)
            except ValueError:
                continue
    return out


class RateLimiter:
    """Sliding-window RPM cap + bounded concurrency. Use as an async context manager."""

    def __init__(
        self, rpm: int | None = None, concurrency: int | None = None, window: float = 60.0
    ):
        self.rpm = rpm
        self.window = window
        self._semaphore = asyncio.Semaphore(concurrency) if concurrency else None
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        if self._semaphore is not None:
            await self._semaphore.acquire()
        if self.rpm is not None:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    while self._stamps and now - self._stamps[0] >= self.window:
                        self._stamps.popleft()
                    if len(self._stamps) < self.rpm:
                        self._stamps.append(now)
                        break
                    wait = self.window - (now - self._stamps[0])
                await asyncio.sleep(max(wait, 0.01))
        return self

    async def __aexit__(self, *exc):
        if self._semaphore is not None:
            self._semaphore.release()


_LIMITERS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, RateLimiter]]" = (
    weakref.WeakKeyDictionary()
)


def get_limiter(model: str) -> RateLimiter:
    """Limiter for the model's provider, scoped to the running event loop."""
    provider = model.split("/", 1)[0]
    loop = asyncio.get_running_loop()
    per_loop = _LIMITERS.setdefault(loop, {})
    if provider not in per_loop:
        per_loop[provider] = RateLimiter(
            rpm=_parse_env_map("PROVIDER_RPM").get(provider),
            concurrency=_parse_env_map("PROVIDER_CONCURRENCY").get(provider),
        )
    return per_loop[provider]
