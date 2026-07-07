"""Evaluator contract: everything scores a stored AgentRun, never a live agent."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agenteval.core.task import Task
from agenteval.core.trace import AgentRun


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float  # 0.0 .. 1.0
    detail: str = ""


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, run: AgentRun, task: Task) -> list[CheckResult]: ...


class Check:
    """Base for single-result checks declared in a Task's `checks` list.

    Subclasses implement `evaluate(run, task) -> CheckResult` (sync, or async
    for LLM-backed checks like the judge — the engine awaits when needed).
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:  # pragma: no cover
        raise NotImplementedError
