"""Raw tool-calling loop adapter — the reference implementation.

We own the loop, so tracing is exact: every LLM call, tool call, and tool
result becomes a Step in the normalized AgentRun.
"""

import inspect
import json
import time
import uuid

from agenteval.core.llm import complete
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun, Step, StepType

SYSTEM_PROMPT = (
    "You are a capable assistant. Use the provided tools whenever they can answer part of "
    "the question, and base your answer on their output. If you cannot answer reliably with "
    "the available tools and information, say you don't know instead of guessing."
)

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}


def tool_schema(fn) -> dict:
    """Build an OpenAI-style tool schema from a plain Python callable."""
    sig = inspect.signature(fn)
    properties, required = {}, []
    for name, param in sig.parameters.items():
        json_type = _TYPE_MAP.get(param.annotation, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or fn.__name__,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
    # Example args let the offline mock model exercise tools deterministically.
    example_args = getattr(fn, "example_args", None)
    if example_args is not None:
        schema["function"]["x_example_args"] = example_args
    return schema


class RawAgent:
    """Minimal agent-under-test: LLM + tools in a loop until a final answer."""

    name = "raw"

    def __init__(self, model: str, max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations

    async def run(self, task: Task) -> AgentRun:
        run = AgentRun(
            task_id=task.id,
            run_id=f"{task.id}:0:{uuid.uuid4().hex[:8]}",
            framework=self.name,
            input=task.input,
            metadata={"model": self.model},
        )
        tools_by_name = {fn.__name__: fn for fn in task.tools}
        schemas = [tool_schema(fn) for fn in task.tools]
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.input["prompt"]},
        ]
        started = time.perf_counter()
        try:
            for _ in range(self.max_iterations):
                response = await complete(self.model, messages, tools=schemas or None)
                run.steps.append(
                    Step(
                        type=StepType.LLM_CALL,
                        content=response.content,
                        model=self.model,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        latency_ms=response.latency_ms,
                        timestamp=time.time(),
                    )
                )
                run.total_tokens += response.tokens_in + response.tokens_out
                run.total_cost_usd += response.cost_usd

                if not response.tool_calls:
                    run.final_output = response.content or ""
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                )
                for tc in response.tool_calls:
                    run.steps.append(
                        Step(
                            type=StepType.TOOL_CALL,
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                            timestamp=time.time(),
                        )
                    )
                    tool_started = time.perf_counter()
                    output = self._execute_tool(tools_by_name, tc.name, tc.arguments)
                    run.steps.append(
                        Step(
                            type=StepType.TOOL_RESULT,
                            tool_name=tc.name,
                            tool_output=output,
                            latency_ms=(time.perf_counter() - tool_started) * 1000,
                            timestamp=time.time(),
                        )
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            else:
                run.error = f"no final answer after {self.max_iterations} iterations"
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
        run.total_latency_ms = (time.perf_counter() - started) * 1000
        return run

    @staticmethod
    def _execute_tool(tools_by_name: dict, name: str, args: dict) -> str:
        fn = tools_by_name.get(name)
        if fn is None:
            return f"ERROR: unknown tool '{name}'"
        try:
            return str(fn(**args))
        except Exception as e:
            # Tool failures go back to the agent as text — recovering from them
            # is part of what the adversarial category measures.
            return f"ERROR: {type(e).__name__}: {e}"
