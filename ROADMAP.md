# AgentEval Roadmap

**Next release (v0.2): multi-agent evaluation.** Trace, run and score systems where several agents hand work to each other, and answer two questions single-agent evals can't: *did the team beat one agent for its cost?* and *when it failed, which agent broke it?*

Each phase keeps the project's rules: zero-key demo still works, evaluators read stored traces only, every new check is unit-tested with synthetic traces.

---

## Phase 0: Housekeeping

- [ ] Stop tracking `.claude/settings.json` (local Claude Code permissions); add `.claude/` to `.gitignore`
- [ ] Re-run CI on current dependency versions; pin anything that broke
- [ ] Verify the CrewAI adapter end-to-end on Python 3.12 *(carried over)*
- [ ] Open one GitHub issue per roadmap item and link each PR to its issue

## Phase 1: Multi-agent trace format

Today `Step` has no idea which agent produced it, so a team run collapses into one flat list.

- [ ] Add `agent: str = "main"` to `Step` and `agents: list[str]` to `AgentRun` (`core/trace.py`). The defaults keep every existing trace and test valid
- [ ] New `StepType.HANDOFF` with `from_agent`, `to_agent` and `content` (the message passed along)
- [ ] Store and load the new fields in `core/store.py`; databases written by v0.1 still load
- [ ] Synthetic fixtures in `tests/fixtures/`: clean handoff, delegation loop, dropped context, wrong specialist
- [ ] Conformance test: every adapter's trace validates against the new schema

## Phase 2: Reference team (the agent under test)

A small supervisor team: just enough to exercise every multi-agent check, still runnable offline.

- [ ] `adapters/raw_team.py`: a supervisor routes to a **researcher** (tools), an **analyst** (calculation) and a **writer**, plus an optional **critic** that can send work back
- [ ] Mock models learn routing decisions, so `mock/good` vs `mock/weak` teams still score differently with zero API keys
- [ ] `adapters/langgraph_team.py`: the same team in LangGraph's supervisor pattern. A conformance test asserts it emits the same trace shape as `raw_team` (as raw vs langgraph does today)
- [ ] CLI: `--framework raw-team | langgraph-team`
- [ ] Update spec §8 ("agents under test are minimal fixtures"): the team stays compact; it is a fixture, not a product

## Phase 3: Multi-agent checks and benchmark category

- [ ] New programmatic checks in `evaluators/programmatic.py`: `HandoffOccurred(from_agent, to_agent)`, `AgentInvoked`, `AgentNotInvoked`, `MaxHandoffs(n)`, `NoDelegationLoop`, `ToolScope(agent, allowed_tools)`
- [ ] Context preservation: a fact from the user's input must reach the agent that needs it
- [ ] Failure attribution: extend faithfulness so each unsupported claim is traced to the agent that first introduced it
- [ ] Metrics: per-agent tokens, cost and latency; **coordination overhead** (share of tokens spent routing rather than working)
- [ ] New `benchmarks/multi_agent/` category with 13 tasks like the others: delegation, merging parallel research, conflicting specialists, a critic loop that must terminate, injection passed through a handoff
- [ ] Headline experiment: run the same tasks with a single agent and with the team, and add a README table answering "is the team worth its cost?"

## Phase 4: Dashboard

- [ ] Swimlane trace view: one lane per agent, arrows for handoffs (`AgentEvalDashboard.jsx`)
- [ ] Per-agent breakdown table and a "blamed agent" column for failed tasks
- [ ] Regression over time across stored runs *(carried over)*

## Phase 5: Make it usable by others

- [ ] OpenAI Agents SDK adapter (it has native handoffs, so it fits naturally after Phase 1)
- [ ] MCP tools: let a task expose tools served by an MCP server
- [ ] GitHub Action: run a suite on a pull request, comment the scorecard, fail on regression against a baseline run
- [ ] Publish to PyPI (check whether the name `agenteval` is free first; choose another if it isn't)

## Phase 6: Adversarial depth

- [ ] Multi-turn injection chains *(carried over)*
- [ ] Cross-agent injection: poisoned tool output reaching the researcher tries to make the writer call a destructive tool

---

## Definition of done for v0.2

- CI green with zero API keys, including a `raw-team` demo run
- Every new check covered by synthetic-trace unit tests
- README updated: multi-agent section, swimlane screenshot, single-agent vs team results table
