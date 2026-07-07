"""CrewAI adapter: runs a single-agent Crew and emits a normalized AgentRun.

CrewAI drives its own executor on top of LiteLLM, so this adapter uses real
provider model strings (the offline mock models can't speak CrewAI's executor
protocol). Tool calls are captured exactly via recording wrappers; LLM usage
is aggregated from CrewAI's usage metrics into a single summary llm_call step
(CrewAI does not expose clean per-call granularity).

Note: as of mid-2026 CrewAI does not install on Python 3.14 (old numpy pin);
use a Python <= 3.12 environment for this adapter.
"""

import asyncio
import functools
import time
import uuid

from agenteval.core.llm import estimate_cost
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun, Step, StepType

GOAL = (
    "Complete the user's task accurately, using the available tools when they can answer part "
    "of the question. If you cannot answer reliably, say you don't know instead of guessing."
)


class CrewAIAgent:
    """Minimal single-agent Crew under test (a fixture, not a product)."""

    name = "crewai"

    def __init__(self, model: str, max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations

    async def run(self, task: Task) -> AgentRun:
        try:
            import crewai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "CrewAI adapter needs crewai (Python <= 3.12 as of mid-2026): pip install crewai"
            ) from e
        # CrewAI is synchronous; keep the runner's event loop free.
        return await asyncio.to_thread(self._run_sync, task)

    def _run_sync(self, task: Task) -> AgentRun:
        from crewai import LLM, Agent, Crew
        from crewai import Task as CrewTask
        from crewai.tools import tool as crew_tool

        run = AgentRun(
            task_id=task.id,
            run_id=f"{task.id}:0:{uuid.uuid4().hex[:8]}",
            framework=self.name,
            input=task.input,
            metadata={"model": self.model},
        )

        def recording(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                run.steps.append(
                    Step(
                        type=StepType.TOOL_CALL,
                        tool_name=fn.__name__,
                        tool_args=kwargs or None,
                        timestamp=time.time(),
                    )
                )
                tool_started = time.perf_counter()
                try:
                    output = str(fn(*args, **kwargs))
                except Exception as e:
                    output = f"ERROR: {type(e).__name__}: {e}"
                run.steps.append(
                    Step(
                        type=StepType.TOOL_RESULT,
                        tool_name=fn.__name__,
                        tool_output=output,
                        latency_ms=(time.perf_counter() - tool_started) * 1000,
                        timestamp=time.time(),
                    )
                )
                return output

            return crew_tool(wrapper)

        agent = Agent(
            role="Assistant",
            goal=GOAL,
            backstory="A careful assistant being evaluated on benchmark tasks.",
            tools=[recording(fn) for fn in task.tools],
            llm=LLM(model=self.model, temperature=0),
            max_iter=self.max_iterations,
            verbose=False,
        )
        crew = Crew(
            agents=[agent],
            tasks=[
                CrewTask(
                    description=task.input["prompt"],
                    expected_output="A direct, complete answer to the user's question.",
                    agent=agent,
                )
            ],
            verbose=False,
        )

        started = time.perf_counter()
        try:
            result = crew.kickoff()
            run.final_output = str(result)
            usage = getattr(crew, "usage_metrics", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
            run.steps.append(
                Step(
                    type=StepType.LLM_CALL,
                    content="(aggregated: CrewAI does not expose per-call LLM traces)",
                    model=self.model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    timestamp=time.time(),
                )
            )
            run.total_tokens = tokens_in + tokens_out
            run.total_cost_usd = estimate_cost(self.model, tokens_in, tokens_out)
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
        run.total_latency_ms = (time.perf_counter() - started) * 1000
        return run
