from agenteval.core.config import DEFAULT_AGENT_MODEL, load_config


def test_defaults_when_env_unset(monkeypatch):
    for var in ("AGENT_MODEL", "JUDGE_MODEL", "WEAK_MODEL", "MAX_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.agent_model == DEFAULT_AGENT_MODEL
    assert cfg.max_concurrency == 4


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "mock/good")
    monkeypatch.setenv("JUDGE_MODEL", "mock/judge")
    monkeypatch.setenv("WEAK_MODEL", "mock/weak")
    monkeypatch.setenv("MAX_CONCURRENCY", "9")
    cfg = load_config()
    assert cfg.agent_model == "mock/good"
    assert cfg.judge_model == "mock/judge"
    assert cfg.weak_model == "mock/weak"
    assert cfg.max_concurrency == 9
