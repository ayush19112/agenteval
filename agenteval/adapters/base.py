"""Adapter protocol: one method that runs a task and emits a normalized AgentRun."""

from typing import Protocol, runtime_checkable

from agenteval.core.task import Task
from agenteval.core.trace import AgentRun


@runtime_checkable
class AgentAdapter(Protocol):
    name: str

    async def run(self, task: Task) -> AgentRun: ...
