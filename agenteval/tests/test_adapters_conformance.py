"""Phase 9: the same task must yield comparable AgentRuns across adapters."""

import importlib.util

import pytest

from agenteval.adapters.raw import RawAgent
from agenteval.core.task import Task, load_benchmarks
from agenteval.core.trace import StepType
from agenteval.evaluators.engine import evaluate_run

needs_langgraph = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
)
needs_crewai = pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None,
    reason="crewai not installed (needs Python <= 3.12)",
)


def sample_tool(city: str) -> str:
    """Look up a city."""
    return f"data for {city}"


sample_tool.example_args = {"city": "Tokyo"}


def make_task() -> Task:
    return Task(
        id="conformance",
        category="tool_use",
        input={"prompt": "What is the data for Tokyo?"},
        tools=[sample_tool],
    )


@needs_langgraph
async def test_langgraph_matches_raw_trace_shape():
    from agenteval.adapters.langgraph import LangGraphAgent

    raw = await RawAgent("mock/good").run(make_task())
    lg = await LangGraphAgent("mock/good").run(make_task())

    assert lg.error is None, lg.error
    assert lg.framework == "langgraph"
    for run in (raw, lg):
        types = [s.type for s in run.steps]
        assert types.count(StepType.TOOL_CALL) == 1
        assert types.count(StepType.TOOL_RESULT) == 1
        assert types.count(StepType.LLM_CALL) >= 2
        assert "data for Tokyo" in run.final_output
        assert run.total_tokens > 0

    tool_call = next(s for s in lg.steps if s.type == StepType.TOOL_CALL)
    assert tool_call.tool_name == "sample_tool"
    assert tool_call.tool_args == {"city": "Tokyo"}


@needs_langgraph
async def test_langgraph_passes_real_benchmark_checks():
    from agenteval.adapters.langgraph import LangGraphAgent

    task = load_benchmarks()["weather_and_currency"]
    run = await LangGraphAgent("mock/good").run(task)
    results = await evaluate_run(run, task, judge_model="mock/judge")
    by_name = {r.name: r for r in results}
    assert by_name["ToolCalled(get_weather)"].passed
    assert by_name["ToolCalled(convert_currency)"].passed
    assert by_name["Contains('15,700')"].passed


@needs_crewai
@pytest.mark.integration  # CrewAI's executor needs a real provider model
async def test_crewai_produces_normalized_run():
    from agenteval.adapters.crewai import CrewAIAgent
    from agenteval.core.config import load_config

    run = await CrewAIAgent(load_config().agent_model).run(make_task())
    assert run.framework == "crewai"
    assert run.error is None, run.error
    assert run.final_output
    assert any(s.type == StepType.TOOL_CALL for s in run.steps)
