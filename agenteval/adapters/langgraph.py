"""LangGraph adapter: runs a minimal prebuilt ReAct agent and emits the same
normalized AgentRun as every other adapter.

A bridge chat model routes LangGraph's LLM calls through core.llm.complete,
so any model string works — including the offline mock models — and each
call's tokens, cost, and latency are captured for the trace. The trace itself
is reconstructed from the graph's final message list, so nothing depends on
LangGraph callbacks.
"""

import functools
import json
import time
import uuid

from agenteval.adapters.raw import SYSTEM_PROMPT
from agenteval.core.llm import complete
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun, Step, StepType


def _to_api_message(message) -> dict:
    kind = message.type
    if kind == "system":
        return {"role": "system", "content": message.content}
    if kind == "human":
        return {"role": "user", "content": message.content}
    if kind == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": str(message.content),
        }
    out: dict = {"role": "assistant", "content": message.content or None}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
            }
            for tc in message.tool_calls
        ]
    return out


def _safe_tool(fn):
    """Match the raw adapter's contract: tool failures return an ERROR string
    for the agent to recover from (the adversarial category tests this),
    instead of crashing the graph."""

    @functools.wraps(fn)  # copies __dict__, so example_args survives
    def wrapper(*args, **kwargs):
        try:
            return str(fn(*args, **kwargs))
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    return wrapper


def _make_bridge(model: str, records: list):
    """Build the LangChain chat-model bridge lazily (langchain_core optional)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class BridgeChatModel(BaseChatModel):
        model_name: str
        tool_schemas: list = []

        @property
        def _llm_type(self) -> str:
            return "agenteval-bridge"

        def bind_tools(self, tools, **kwargs):
            schemas = []
            for t in tools:
                schema = convert_to_openai_tool(t)
                # surface example args so the offline mock model can drive tools
                fn = getattr(t, "func", None)
                example_args = getattr(fn, "example_args", None)
                if example_args is not None:
                    schema["function"]["x_example_args"] = example_args
                schemas.append(schema)
            return self.model_copy(update={"tool_schemas": schemas})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise NotImplementedError("use ainvoke — this bridge is async-only")

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            response = await complete(
                self.model_name,
                [_to_api_message(m) for m in messages],
                tools=self.tool_schemas or None,
            )
            # closure, not a pydantic field: LangGraph deep-copies the model,
            # which would silently detach a field-held list
            records.append(response)
            ai = AIMessage(
                content=response.content or "",
                tool_calls=[
                    {"name": tc.name, "args": tc.arguments, "id": tc.id}
                    for tc in response.tool_calls
                ],
                usage_metadata={
                    "input_tokens": response.tokens_in,
                    "output_tokens": response.tokens_out,
                    "total_tokens": response.tokens_in + response.tokens_out,
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])

    return BridgeChatModel(model_name=model)


class LangGraphAgent:
    """Minimal LangGraph ReAct agent-under-test (a fixture, not a product)."""

    name = "langgraph"

    def __init__(self, model: str, max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations

    async def run(self, task: Task) -> AgentRun:
        try:
            from langchain_core.tools import StructuredTool
            from langgraph.prebuilt import create_react_agent
        except ImportError as e:
            raise RuntimeError(
                "LangGraph adapter needs langgraph + langchain-core: "
                "pip install langgraph langchain-core"
            ) from e

        run = AgentRun(
            task_id=task.id,
            run_id=f"{task.id}:0:{uuid.uuid4().hex[:8]}",
            framework=self.name,
            input=task.input,
            metadata={"model": self.model},
        )
        records: list = []
        bridge = _make_bridge(self.model, records)
        lc_tools = [StructuredTool.from_function(_safe_tool(fn)) for fn in task.tools]
        graph = create_react_agent(bridge, lc_tools, prompt=SYSTEM_PROMPT)

        started = time.perf_counter()
        try:
            state = await graph.ainvoke(
                {"messages": [("user", task.input["prompt"])]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            llm_index = 0
            for message in state["messages"]:
                if message.type == "ai":
                    record = records[llm_index] if llm_index < len(records) else None
                    llm_index += 1
                    usage = getattr(message, "usage_metadata", None) or {}
                    run.steps.append(
                        Step(
                            type=StepType.LLM_CALL,
                            content=message.content or None,
                            model=self.model,
                            tokens_in=record.tokens_in if record else usage.get("input_tokens", 0),
                            tokens_out=(
                                record.tokens_out if record else usage.get("output_tokens", 0)
                            ),
                            latency_ms=record.latency_ms if record else 0.0,
                            timestamp=time.time(),
                        )
                    )
                    for tc in message.tool_calls:
                        run.steps.append(
                            Step(
                                type=StepType.TOOL_CALL,
                                tool_name=tc["name"],
                                tool_args=tc["args"],
                                timestamp=time.time(),
                            )
                        )
                elif message.type == "tool":
                    run.steps.append(
                        Step(
                            type=StepType.TOOL_RESULT,
                            tool_name=message.name,
                            tool_output=str(message.content),
                            timestamp=time.time(),
                        )
                    )
            final = state["messages"][-1]
            run.final_output = final.content if final.type == "ai" else ""
            run.total_tokens = sum(r.tokens_in + r.tokens_out for r in records)
            run.total_cost_usd = sum(r.cost_usd for r in records)
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
        run.total_latency_ms = (time.perf_counter() - started) * 1000
        return run
