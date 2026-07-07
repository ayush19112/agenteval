"""Async execution: runs each task `samples` times through an adapter.

- bounded semaphore caps concurrent agent runs (each run is mostly LLM I/O)
- resumable: samples already persisted for (task, framework, model) are skipped
- provider-level retry/backoff lives in core.llm, not here
"""

import asyncio
import uuid

from agenteval.adapters.base import AgentAdapter
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun


async def run_task(adapter: AgentAdapter, task: Task, sample: int = 0) -> AgentRun:
    run = await adapter.run(task)
    run.run_id = f"{task.id}:{sample}:{uuid.uuid4().hex[:8]}"
    run.metadata["sample"] = sample
    return run


async def run_suite(
    adapter: AgentAdapter,
    tasks: list[Task],
    store=None,
    samples: int | None = None,
    max_concurrency: int = 4,
    resume: bool = True,
) -> list[AgentRun]:
    """Run every task; returns only the runs executed now (resumed ones are skipped)."""
    semaphore = asyncio.Semaphore(max_concurrency)
    model = getattr(adapter, "model", "")

    jobs: list[tuple[Task, int]] = []
    for task in tasks:
        n = samples if samples is not None else task.samples
        done: set[int] = set()
        if store is not None and resume:
            done = store.completed_samples(task.id, adapter.name, model)
        jobs.extend((task, i) for i in range(n) if i not in done)

    async def _one(task: Task, sample: int) -> AgentRun:
        async with semaphore:
            run = await run_task(adapter, task, sample)
        if store is not None:
            store.save_run(run, category=task.category, model=model)
        return run

    return list(await asyncio.gather(*(_one(t, i) for t, i in jobs)))
