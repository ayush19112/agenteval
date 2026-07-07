"""The raw adapter driven by the offline mock model — no network, no keys."""

from agenteval.adapters.raw import RawAgent, tool_schema
from agenteval.core.task import Task
from agenteval.core.trace import StepType


def sample_tool(city: str) -> str:
    """Look up a city."""
    return f"data for {city}"


sample_tool.example_args = {"city": "Tokyo"}


def make_task(tools=None) -> Task:
    return Task(
        id="demo",
        category="tool_use",
        input={"prompt": "What is the data for Tokyo?"},
        tools=tools if tools is not None else [sample_tool],
    )


def test_tool_schema_from_callable():
    schema = tool_schema(sample_tool)
    fn = schema["function"]
    assert fn["name"] == "sample_tool"
    assert fn["parameters"]["properties"]["city"]["type"] == "string"
    assert fn["parameters"]["required"] == ["city"]
    assert fn["x_example_args"] == {"city": "Tokyo"}


async def test_good_mock_uses_tools_and_traces_everything():
    run = await RawAgent("mock/good").run(make_task())
    types = [s.type for s in run.steps]
    assert types == [StepType.LLM_CALL, StepType.TOOL_CALL, StepType.TOOL_RESULT, StepType.LLM_CALL]
    assert run.steps[1].tool_args == {"city": "Tokyo"}
    assert "data for Tokyo" in run.final_output
    assert run.total_tokens > 0
    assert run.total_latency_ms > 0
    assert run.error is None


async def test_weak_mock_skips_tools():
    run = await RawAgent("mock/weak").run(make_task())
    assert all(s.type == StepType.LLM_CALL for s in run.steps)
    assert run.final_output  # answers confidently...
    assert "data for Tokyo" not in run.final_output  # ...without any evidence


async def test_good_mock_abstains_without_tools():
    run = await RawAgent("mock/good").run(make_task(tools=[]))
    assert "don't know" in run.final_output.lower()


async def test_tool_error_is_returned_to_agent_not_raised():
    def broken(city: str) -> str:
        """Always fails."""
        raise ValueError("boom")

    broken.example_args = {"city": "Tokyo"}
    run = await RawAgent("mock/good").run(make_task(tools=[broken]))
    tool_results = [s for s in run.steps if s.type == StepType.TOOL_RESULT]
    assert tool_results and tool_results[0].tool_output.startswith("ERROR:")
    assert run.error is None
