# Agent Evaluation Platform — Spec & Build Plan

A framework-agnostic **"pytest for agents"**: point it at any agent (LangGraph, CrewAI, or a raw tool-calling loop), run a benchmark suite, and get a scorecard across **correctness, cost, latency, and behavior**.

This is a portfolio/resume project. The goal is a **small, finished, demoable system** with sound design decisions — not a production platform. Prefer finishing a tight MVP over building every feature.

---

## How to use this document (for Claude Code)

- Build in **phase order** (Phase 0 → 11). Each phase has a **deliverable** and **acceptance criteria**. Do not start a phase until the previous one meets its criteria.
- Phases 0–5 are the **MVP**. Everything after is an extension.
- The **data models** in this doc are the contract everything depends on. Do not change their shape without asking; downstream code assumes them.
- Write **unit tests using synthetic (hand-written) `AgentRun` objects** — most of the system is testable with no LLM calls. See "Testing strategy."
- Keep model names in **config/env vars**, never hardcoded (free API tiers change often).
- Ask before adding dependencies beyond those listed, or before building anything under "Non-goals."

---

## 1. Core design principle

**Separate execution from evaluation, connected by one normalized trace format.**

1. An **adapter** runs the agent (slow, costs tokens, non-deterministic) and emits a normalized `AgentRun`.
2. The `AgentRun` is **persisted** to a store.
3. **Evaluators** read stored `AgentRun`s and produce scores (fast, free, repeatable).

This means you can add or re-tune an evaluator and **re-score without re-running the agent**. Everything downstream reads the normalized trace only — never framework internals. That is what makes the harness framework-agnostic.

---

## 2. Tech stack & constraints

**Everything must be free.** Only LLM inference can cost money, and that is covered by free tiers / local models.

- **Language:** Python 3.11+
- **Models:** all calls go through **LiteLLM** (one interface for every provider; also gives token/cost accounting). Defaults, all swappable via env vars:
  - Agent under test: Groq (fast, free) or local Ollama
  - Judge: Google Gemini Flash (free tier, higher quality for grading)
  - Deliberately "weak" agent: a small local Ollama model
- **Async:** `asyncio` (work is I/O-bound)
- **Retries:** `tenacity` (exponential backoff + jitter on rate-limit/5xx)
- **Data models:** `pydantic` v2
- **Storage:** **SQLite** (zero-config, file-based; no Postgres, no cloud DB)
- **Static report:** Jinja2 (MVP)
- **Dashboard (later):** FastAPI + React (Vite) + Recharts
- **Tests:** pytest
- **Agent frameworks:** LangGraph, CrewAI (both open-source)
- **Hosting:** GitHub (repo), GitHub Pages / Vercel free tier (dashboard)

**Do not** depend on any API that requires a credit card or has no free tier.

---

## 3. Repository structure

```
agenteval/
  core/
    trace.py          # AgentRun, Step
    task.py           # Task, @task decorator, Category
    runner.py         # async execution, sampling (pass@k), concurrency, retries, resume
    store.py          # SQLite persistence, resume
    config.py         # env-var driven model/provider config
  adapters/
    base.py           # AgentAdapter protocol
    raw.py            # raw tool-calling loop  (BUILD FIRST)
    langgraph.py      # LangGraph adapter
    crewai.py         # CrewAI adapter
  evaluators/
    base.py           # Evaluator protocol, Check, CheckResult
    programmatic.py   # ExactMatch, Contains, Regex, ToolCalled, ...
    judge.py          # calibrated LLM-as-judge
    hallucination.py  # faithfulness + abstention
    metrics.py        # cost, latency, pass@k aggregation
  benchmarks/
    tool_use/         # task modules
    reasoning/
    long_horizon/
    adversarial/
  calibration/
    golden_set.jsonl  # human-scored runs for judge calibration
    calibrate.py      # computes judge/human agreement (Cohen's kappa)
  report/
    static.py         # Jinja2 HTML scorecard (MVP)
    dashboard/        # FastAPI + React (later phase)
  tests/
    fixtures/         # hand-written synthetic AgentRun traces
    test_*.py
  cli.py              # `run` / `evaluate` / `report`
  pytest_plugin.py    # discovers @task fns, runs via core, pass/fail on threshold
pyproject.toml
README.md
```

---

## 4. Data models (the contract)

```python
# core/trace.py
from enum import Enum
from pydantic import BaseModel, Field

class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

class Step(BaseModel):
    type: StepType
    content: str | None = None
    # llm_call fields
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    # tool fields
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_output: str | None = None
    latency_ms: float = 0.0
    timestamp: float = 0.0

class AgentRun(BaseModel):
    task_id: str
    run_id: str                      # unique per sample: f"{task_id}:{sample}:{uuid4}"
    framework: str                   # "raw" | "langgraph" | "crewai"
    input: dict
    steps: list[Step] = Field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None
    metadata: dict = Field(default_factory=dict)
```

```python
# core/task.py
from typing import Literal
from pydantic import BaseModel, Field

Category = Literal["tool_use", "reasoning", "long_horizon", "adversarial"]

class Task(BaseModel):
    id: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    input: dict                      # e.g. {"prompt": "..."}
    tools: list = Field(default_factory=list)   # tool callables for the sandbox
    checks: list = Field(default_factory=list)  # programmatic + judge checks
    samples: int = 5                 # runs per task (for pass@k)
    reference: str | None = None     # gold answer, used by judge / faithfulness
```

```python
# evaluators/base.py
from typing import Protocol
from pydantic import BaseModel
from agenteval.core.trace import AgentRun
from agenteval.core.task import Task

class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float           # 0.0 .. 1.0
    detail: str = ""

class Evaluator(Protocol):
    def evaluate(self, run: AgentRun, task: Task) -> list[CheckResult]: ...
```

```python
# adapters/base.py
from typing import Protocol
from agenteval.core.trace import AgentRun
from agenteval.core.task import Task

class AgentAdapter(Protocol):
    name: str
    async def run(self, task: Task) -> AgentRun: ...
```

**Task declaration** (pytest-style — how a benchmark task is written):

```python
# benchmarks/tool_use/weather_currency.py
from agenteval.core.task import task
from agenteval.evaluators.programmatic import ToolCalled, Contains
from agenteval.evaluators.judge import Judge
from .mocks import mock_weather, mock_currency

@task(category="tool_use", difficulty="medium", samples=5)
def weather_and_currency():
    return dict(
        input={"prompt": "What's the weather in Tokyo, and how much is 100 USD in JPY?"},
        tools=[mock_weather, mock_currency],
        reference="~15C in Tokyo and roughly 15,700 JPY",
        checks=[
            ToolCalled("get_weather", args={"city": "Tokyo"}),
            ToolCalled("convert_currency"),
            Judge("Does the answer give both the weather and a JPY amount?"),
        ],
    )
```

---

## 5. Component specs

### 5.1 Adapters
Each adapter turns one framework's execution into one `AgentRun`. Build **`raw.py` first** — you control the loop, so tracing is trivial.

Raw agent loop:
```
send {prompt + tool schemas} to the LLM (via LiteLLM)
  while the LLM requests a tool:
      run the tool, append TOOL_CALL + TOOL_RESULT steps
      send the tool result back
  record final answer + tokens + latency into the AgentRun
```
LangGraph/CrewAI adapters (later) hook each framework's callbacks/streaming to capture the same steps. Keep the wrapped agents **minimal** — they are test fixtures, not the product.

### 5.2 Runner (`core/runner.py`)
- Runs each task `samples` times (pass@k needs multiple samples — agents are non-deterministic).
- `asyncio` with a **bounded semaphore** capping concurrent LLM calls.
- Retries with **exponential backoff + jitter** on rate-limit (429) / 5xx via `tenacity`.
- **Resumable:** checkpoint completed `run_id`s to the store; re-running skips them.

### 5.3 Trace store (`core/store.py`)
SQLite, two tables:
- `runs(run_id PK, task_id, framework, model, category, blob JSON, created_at)`
- `results(run_id FK, check_name, passed, score, detail)`

The store is the pivot: `run` populates `runs`; `evaluate` reads `runs` and writes `results`; `report` reads both. Re-scoring never touches the agent.

### 5.4 Evaluators (`evaluators/`)
Prefer deterministic checks; use the judge only for open-ended quality.

**Programmatic (`programmatic.py`):**
- `ExactMatch`, `Contains`, `Regex`, `JSONSchema`
- `ToolCalled(name, args=None, order=None)`, `ToolNotCalled(name)` (for adversarial)
- `StepCountMax(n)` (long-horizon efficiency)
- `OutputNonEmpty`

**Calibrated LLM judge (`judge.py`):**
- Rubric + optional `reference`; **structured JSON output**; temperature 0; majority vote across a few samples on borderline scores.
- **Calibration is a first-class feature:** `calibration/golden_set.jsonl` holds ~30–50 human-scored runs. `calibrate.py` runs the judge over them and computes **Cohen's kappa** vs. the human labels. Tune the judge prompt until **kappa >= 0.6**, and **report kappa on the scorecard** so every judge number ships with a known reliability.

**Hallucination (`hallucination.py`):** two measures, both LLM-judged (so subject to the same calibration discipline).
- *Faithfulness:* decompose `final_output` into atomic claims, check each against the evidence in the trace (`tool_output`s + `reference`). Score = supported / total.
- *Abstention:* on adversarial "unanswerable" tasks (nonexistent entity, missing tool), correct behavior is refusal / "I don't know." Measure correct-abstention rate; a confident fabricated answer counts as a hallucination.

**Metrics (`metrics.py`):**
- *Cost:* use `litellm.completion_cost()` where available; fallback pricing table `model -> ($/1M in, $/1M out)`; local models = $0. Report per-run + p50/p95.
- *Latency:* per-run wall clock + per-step; report p50/p95/p99; split LLM vs. tool time.
- *Success (required):* run k samples/task, report **pass@1** (mean), **pass@k** (any succeeds), **pass^k** (all succeed), plus **per-task variance** to surface flaky tasks. Single-run success is misleading — do not report it alone.

### 5.5 Scorecard / aggregation
Combine per-run `CheckResult`s into per-task → per-category → per-run aggregates: success rates, cost, latency, hallucination/faithfulness, and judge kappa. Support comparing runs (framework A vs B, model X vs Y).

### 5.6 Reporting
- **MVP:** `report/static.py` renders a Jinja2 HTML scorecard — summary, per-category breakdown, failed-task list with trace drill-down.
- **Later:** FastAPI serving SQLite + React (Vite) + Recharts dashboard: scorecard summary, framework/model comparison, trace viewer (the step tree), regression-over-time, failure inspector. This is the showpiece for a frontend portfolio.

### 5.7 CLI (`cli.py`)
`agenteval run --suite benchmarks --framework raw` · `agenteval evaluate` · `agenteval report --out report.html`

### 5.8 pytest plugin (`pytest_plugin.py`)
Discovers `@task` functions, runs them through the core (samples configurable, default 1 in pytest mode for speed), asserts pass/fail against a score threshold, and supports `--eval-report` to emit the scorecard. This is what makes "pytest for agents" literal.

---

## 6. Testing strategy

**Most of the system is tested with no LLM calls.** This is the core testing insight — bake it in.

1. **Unit tests with synthetic traces (bulk of tests, no LLM):** hand-write `AgentRun` fixtures and assert evaluators behave. Keep fixtures in `tests/fixtures/`:
   - a clean successful run,
   - a run that skips a required tool (should fail `ToolCalled`),
   - a run with a fabricated claim not in any tool output (faithfulness should flag it),
   - a run that correctly abstains on an unanswerable task.
   Also unit-test cost math and pass@k aggregation. Fast, deterministic, free.
2. **Integration tests (marked, skippable):** run the raw agent on ~3 tasks against a local Ollama model. Mark `@pytest.mark.integration` so CI can skip them (no model in CI).
3. **Judge calibration test:** kappa on the golden set must clear the threshold.
4. **End-to-end validation (the money shot):** run a **good agent** (bigger model) and a **weak agent** (small model / no tools) on the suite. Assert the scorecard rates good > weak. If it does, the platform works.

---

## 7. Build plan (phases)

### Phase 0 — Project setup
Repo, `pyproject.toml`, dependencies, `core/config.py` (env-var model selection), pre-commit (ruff + black), empty test run.
**Done:** `pip install -e .` works; `pytest` runs (green, empty); config reads `AGENT_MODEL` / `JUDGE_MODEL` / `WEAK_MODEL` from env.

### Phase 1 — Core contract + vertical slice
`trace.py`, `task.py`, `adapters/base.py`, `adapters/raw.py`, a minimal `runner` that runs ONE task ONCE, `cli.py run` that prints the resulting `AgentRun`.
**Done:** a real raw agent (Ollama or Groq) completes one tool-use task and returns a populated `AgentRun` (steps, tokens, latency, final output).

### Phase 2 — Storage + run/evaluate split
`store.py` (SQLite); `run` persists runs; `evaluate` reads runs and scores them; resume skips existing `run_id`s.
**Done:** you can `run` once, then `evaluate` repeatedly **without re-running the agent**.

### Phase 3 — Programmatic checks + first tasks
All programmatic checks with unit tests over synthetic traces; ~12 tasks (3 per category).
**Done:** `evaluate` produces per-task pass/fail; all check unit tests pass.

### Phase 4 — Calibrated LLM judge
`judge.py`, `golden_set.jsonl` (~30 hand-scored), `calibrate.py` (kappa), judge wired into `evaluate`.
**Done:** judge runs; kappa is reported; a calibration test passes at the threshold.

### Phase 5 — Metrics + static scorecard  ← MVP BOUNDARY
`metrics.py` (cost/latency/pass@k), aggregation, `report/static.py` (Jinja2 HTML).
**Done:** `report` produces an HTML scorecard with success rate, cost, and latency by category for the raw agent on ~15 tasks.

### Phase 6 — Hallucination
`hallucination.py` (faithfulness + abstention); adversarial unanswerable tasks.
**Done:** a synthetic hallucinated trace is flagged; a correct-abstention trace scores as abstained.

### Phase 7 — Concurrency + resilience
Async runner with semaphore + retry/backoff; per-provider rate-limit config.
**Done:** a full-suite run completes without crashing on rate limits and resumes cleanly after interruption.

### Phase 8 — Scale benchmark to 50+
Expand tasks across all four categories; document the category balance.
**Done:** suite has >= 50 tasks with a documented spread across categories.

### Phase 9 — Framework adapters
`langgraph.py`, `crewai.py` wrapping simple agents on the same tasks; conformance tests (same trace shape across adapters).
**Done:** the same task runs through all three adapters and yields comparable `AgentRun`s.

### Phase 10 — React dashboard
FastAPI + React (Vite) + Recharts reading SQLite: scorecard, A-vs-B comparison, trace viewer, failure inspector.
**Done:** dashboard supports framework/model comparison and trace drill-down from real data.

### Phase 11 — pytest plugin + CI
`pytest_plugin.py`; CI workflow running unit tests (integration skipped).
**Done:** `pytest` discovers `@task` tests and passes/fails on threshold; CI is green.

---

## 8. Non-goals (do NOT build)

- Authentication, multi-tenancy, or user accounts
- Cloud/hosted database (SQLite only)
- Model training or fine-tuning
- Support for paid-only APIs
- A general-purpose observability platform (this is an *eval* harness)
- Sophisticated agents-under-test — they are minimal fixtures

---

## 9. Definition of done (MVP)

- Phases 0–5 complete.
- One framework (raw), ~15 tasks across all four categories.
- Calibrated judge with reported kappa.
- Static HTML scorecard (success rate, cost, latency, hallucination signal).
- Unit tests green (evaluators tested via synthetic traces, no LLM).
- README with the architecture diagram, a dashboard/scorecard screenshot, and a sample scorecard.

---

## Appendix A — Free model config

Use LiteLLM model strings; keep them in env vars (free tiers and model names change often — never hardcode).

```python
# core/config.py
import os
AGENT_MODEL = os.getenv("AGENT_MODEL", "groq/llama-3.3-70b-versatile")  # fast + free
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")       # free tier, good for grading
WEAK_MODEL  = os.getenv("WEAK_MODEL",  "ollama/llama3.2:3b")            # local, deliberately weak
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
```

- **Ollama** (local, free, no key, no rate limits) is the safe fallback for any model.
- **Google AI Studio (Gemini Flash)** and **Groq** both issue keys with no credit card.
- Build a small fallback so a rate-limit error retries via a different provider or Ollama rather than crashing.

## Appendix B — Sample synthetic trace (for a unit test)

```python
# tests/fixtures/skips_tool.py
from agenteval.core.trace import AgentRun, Step, StepType

# An agent that answered without calling the required weather tool.
skips_weather_tool = AgentRun(
    task_id="weather_and_currency",
    run_id="weather_and_currency:0:test",
    framework="raw",
    input={"prompt": "What's the weather in Tokyo, and 100 USD in JPY?"},
    steps=[
        Step(type=StepType.LLM_CALL, model="test", tokens_in=40, tokens_out=20),
        Step(type=StepType.TOOL_CALL, tool_name="convert_currency", tool_args={"amount": 100}),
        Step(type=StepType.TOOL_RESULT, tool_name="convert_currency", tool_output="15700 JPY"),
    ],
    final_output="It's sunny in Tokyo, and 100 USD is about 15,700 JPY.",  # weather is fabricated
    total_tokens=60,
)
# Expected: ToolCalled("get_weather") -> FAIL; faithfulness flags the weather claim.
```
