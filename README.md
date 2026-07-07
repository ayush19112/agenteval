# agenteval — "pytest for agents"

A framework-agnostic evaluation harness for LLM agents: point it at any agent (raw tool-calling loop, LangGraph, CrewAI), run a benchmark suite, and get a scorecard across **correctness, cost, latency, and behavior** — including hallucination measurement and a **calibrated** LLM-as-judge whose reliability (Cohen's kappa vs. human labels) is reported right on the scorecard.

![Dashboard](docs/dashboard-light.png)

*The interactive dashboard on real data: the 52-task suite run through three configurations. The raw and LangGraph adapters wrap the **same** agent behavior and score identically on every category — that's the framework-agnostic trace contract doing its job — while the deliberately weak agent fails on every axis (pass@k, judge score, faithfulness, correct abstention).*

## Why this exists

Agent quality is stochastic, multi-dimensional, and framework-fragmented. Most eval setups re-run the agent every time a metric changes, trust an uncalibrated judge, and report single-run success. This project takes the opposite positions:

1. **Execution is separated from evaluation** by one normalized trace format (`AgentRun`). The agent runs once (slow, costs tokens); every evaluator reads the stored trace (fast, free, repeatable). Add or re-tune an evaluator and re-score **without re-running the agent**.
2. **The judge is calibrated, not trusted.** A golden set of human-labeled answers measures judge/human agreement (Cohen's kappa). Every judge number on the scorecard ships with its known reliability.
3. **pass@k, not pass-once.** Every task runs k samples; the scorecard reports pass@1 / pass@k / pass^k and per-task variance to surface flaky behavior.

## Architecture

```
            adapters (raw | langgraph | crewai)          evaluators (read traces only)
  Task ──▶  run the agent, capture every step  ──▶  AgentRun ──▶ SQLite ──▶  programmatic checks
                                                    (normalized      │       calibrated LLM judge
                                                     trace)          │       faithfulness / abstention
                                                                     ▼       cost / latency / pass@k
                                                              `evaluate` ──▶ results ──▶ scorecard (HTML)
```

- `agenteval run` executes agents and persists traces (resumable, async, bounded concurrency)
- `agenteval evaluate` scores stored traces — repeatable, never touches the agent
- `agenteval report` renders a static HTML scorecard
- `agenteval dashboard` serves the interactive React dashboard (below)

## Quickstart — zero API keys

The built-in deterministic mock models (`mock/good`, `mock/weak`, `mock/judge`) exercise the entire pipeline offline:

```bash
pip install -e ".[dev]"
pytest                                                  # 55 unit tests, no LLM calls

agenteval run --model mock/good  --db demo.db           # 52 tasks x 5 samples
agenteval run --model mock/weak  --db demo.db
agenteval run --framework langgraph --model mock/good --db demo.db
agenteval evaluate --judge-model mock/judge --db demo.db
python -m agenteval.calibration.calibrate --model mock/judge
agenteval report --db demo.db --out report.html
```

Open `report.html` — the good agent must outscore the weak one on every axis. That comparison is the platform's own end-to-end validation.

## Interactive dashboard

FastAPI serving SQLite + React (Vite) + Recharts. KPI cards, framework comparison, per-category breakdown, a task table with **execution-trace waterfall** and per-check drill-down, an **A-vs-B compare mode**, pass@k trend, and a dark theme.

```bash
pip install -e ".[dashboard]"
cd agenteval/report/dashboard/frontend && npm install && npm run build && cd -
agenteval dashboard --db demo.db          # -> http://127.0.0.1:8501
```

| Trace drill-down | Compare mode (dark) |
|---|---|
| ![Trace](docs/dashboard-trace.png) | ![Compare](docs/dashboard-compare-dark.png) |

For frontend development: `agenteval dashboard` in one terminal (API), `npm run dev` in the frontend directory (Vite proxies `/api`).

## Using real (free) models

All model strings are [LiteLLM](https://github.com/BerriAI/litellm) format and live in env vars — free tiers change often, so nothing is hardcoded:

```bash
pip install -e ".[llm]"

set AGENT_MODEL=groq/llama-3.3-70b-versatile   # free tier, fast (needs GROQ_API_KEY)
set JUDGE_MODEL=gemini/gemini-2.5-flash        # free tier (needs GEMINI_API_KEY)
set WEAK_MODEL=ollama/llama3.2:3b              # local via Ollama, no key at all

agenteval run --db real.db
agenteval evaluate --db real.db
python -m agenteval.calibration.calibrate --model %JUDGE_MODEL%
agenteval report --db real.db
```

Groq and Google AI Studio both issue keys with no credit card; Ollama is fully local. Resilience is layered: exponential backoff + jitter on transient errors, per-provider rate limits, an optional cross-provider fallback, and checkpointed samples so an interrupted suite resumes where it left off.

```bash
set PROVIDER_RPM=groq=28,gemini=14          # requests-per-minute caps per provider
set PROVIDER_CONCURRENCY=groq=2,gemini=4    # in-flight request caps per provider
set FALLBACK_MODEL=ollama/llama3.2:3b       # last resort when a provider's retries run out
```

## Framework adapters

Every adapter emits the same normalized `AgentRun`, so evaluators never see framework internals:

- **raw** — a minimal tool-calling loop; the reference implementation.
- **langgraph** — a prebuilt LangGraph ReAct agent. A bridge chat model routes LangGraph's LLM calls through the same LLM client, so even the offline mock models work, and conformance tests assert the trace shape matches raw. On the demo suite, raw and LangGraph score identically — same agent, same measurement, different framework.
- **crewai** — a single-agent Crew with recording tool wrappers. CrewAI drives its own executor, so this adapter needs a real provider model, and (as of mid-2026) CrewAI requires Python ≤ 3.12; its tests skip automatically when it isn't installed.

## Benchmark tasks as pytest tests

The pytest plugin makes "pytest for agents" literal — every `@task` function collects as a test that fails below a score threshold:

```bash
pytest --pyargs agenteval.benchmarks --eval-model mock/good --eval-judge-model mock/judge \
       --eval-samples 3 --eval-threshold 0.7 --eval-report scorecard.html
```

## Benchmark suite

Tasks are declared pytest-style — a decorated function returning inputs, sandbox tools, a reference answer, and checks:

```python
@task(category="tool_use", difficulty="medium", samples=5)
def weather_and_currency():
    return dict(
        input={"prompt": "What's the weather in Tokyo, and how much is 100 USD in JPY?"},
        tools=[get_weather, convert_currency],
        reference="About 15°C and partly cloudy in Tokyo; 100 USD is roughly 15,700 JPY.",
        checks=[
            ToolCalled("get_weather", args={"city": "Tokyo"}),
            ToolCalled("convert_currency"),
            Contains("15,700"),
            Judge("Does the answer give both the Tokyo weather and a JPY amount?"),
            Faithfulness(),
        ],
    )
```

The suite has **52 tasks, 13 per category**, each with a reference answer and 2–6 checks. The four categories stress different failure modes:

| Category | Tasks | Stresses | Example |
|---|---|---|---|
| `tool_use` | 13 | right tool, right args, grounded answer | weather + currency conversion |
| `reasoning` | 13 | multi-hop chaining, arithmetic | Best Picture director → other films |
| `long_horizon` | 13 | dependent multi-step workflows, efficiency (`StepCountMax`) | inventory check → order → status confirmation |
| `adversarial` | 13 | prompt injection in tool output, destructive-tool bait, unanswerable questions, broken tools | injected email, `delete_all_records` trap, fictional city, failing DB |

## Scoring

- **Programmatic checks** (preferred wherever possible): `ExactMatch`, `Contains`, `NotContains`, `Regex`, `JSONSchema`, `ToolCalled(name, args, order)`, `ToolNotCalled`, `StepCountMax`, `OutputNonEmpty`.
- **Calibrated judge**: structured JSON verdicts at temperature 0, always graded against a reference, majority vote on borderline scores. `calibration/golden_set.jsonl` holds human-labeled answers; `calibrate.py` computes Cohen's kappa and the scorecard displays it (threshold: 0.6).
- **Hallucination**: *faithfulness* decomposes the answer into atomic claims and checks each against trace evidence (tool outputs + reference); *abstention* verifies the agent refuses on unanswerable tasks instead of fabricating.
- **Cost & latency**: per-run cost via LiteLLM (fallback pricing table; local models are $0), latency p50/p95/p99 with LLM-vs-tool time split.

## Testing philosophy

Most of the platform is tested with **zero LLM calls**: hand-written synthetic `AgentRun` traces (a clean run, a skipped tool, a fabricated claim, a correct abstention) assert that every evaluator reacts correctly (`agenteval/tests/fixtures/synthetic.py`). The mock model family then exercises the full pipeline end-to-end, and the good-vs-weak comparison validates the platform itself. Integration tests against real models are marked `integration` and skipped by default.

## Project structure

```
agenteval/
  core/            trace (AgentRun/Step), task registry, async runner, SQLite store,
                   LLM client, per-provider rate limiting, config
  adapters/        raw tool-calling loop, LangGraph, CrewAI
  evaluators/      programmatic checks, calibrated judge, hallucination, metrics, engine
  benchmarks/      52 tasks: tool_use / reasoning / long_horizon / adversarial
  calibration/     golden_set.jsonl + Cohen's kappa harness
  report/          Jinja2 HTML scorecard + FastAPI/React dashboard
  tests/           synthetic-trace unit tests
  pytest_plugin.py @task discovery, score thresholds, --eval-report
.github/workflows/ CI: lint + unit tests + zero-key demo + plugin smoke + frontend build
```

## Roadmap

- Dashboard: regression-over-time across multiple stored runs
- Verify the CrewAI adapter end-to-end on a Python 3.12 environment
- Grow the adversarial category with multi-turn injection chains
