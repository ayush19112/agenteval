"""Env-var driven model/provider configuration.

Free tiers and model names change often — model strings live in env vars,
never in code. All model strings are LiteLLM-format ("provider/model"),
plus the built-in offline providers "mock/good" and "mock/weak".
"""

import os
from dataclasses import dataclass

DEFAULT_AGENT_MODEL = "groq/llama-3.3-70b-versatile"  # fast + free tier
DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"  # free tier, good for grading
DEFAULT_WEAK_MODEL = "ollama/llama3.2:3b"  # local, deliberately weak


@dataclass(frozen=True)
class Config:
    agent_model: str
    judge_model: str
    weak_model: str
    max_concurrency: int
    db_path: str


def load_config() -> Config:
    """Read configuration from the environment at call time (not import time)."""
    return Config(
        agent_model=os.getenv("AGENT_MODEL", DEFAULT_AGENT_MODEL),
        judge_model=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        weak_model=os.getenv("WEAK_MODEL", DEFAULT_WEAK_MODEL),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "4")),
        db_path=os.getenv("AGENTEVAL_DB", "agenteval.db"),
    )
