from agenteval.core.trace import AgentRun, Step, StepType


def test_agent_run_roundtrip():
    run = AgentRun(
        task_id="t",
        run_id="t:0:abc",
        framework="raw",
        input={"prompt": "hi"},
        steps=[
            Step(type=StepType.LLM_CALL, model="m", tokens_in=10, tokens_out=5),
            Step(type=StepType.TOOL_CALL, tool_name="f", tool_args={"x": 1}),
            Step(type=StepType.TOOL_RESULT, tool_name="f", tool_output="42"),
        ],
        final_output="42",
        total_tokens=15,
    )
    restored = AgentRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.steps[1].tool_args == {"x": 1}
