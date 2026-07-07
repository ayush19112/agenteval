"""Applies a task's checks to stored runs and writes results back to the store.

Pure function over persisted AgentRuns — re-scoring never re-runs an agent.
"""

import asyncio
import inspect

from agenteval.core.task import Task
from agenteval.core.trace import AgentRun
from agenteval.evaluators.base import CheckResult


async def evaluate_run(
    run: AgentRun, task: Task, judge_model: str | None = None
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in task.checks:
        # LLM-backed checks (Judge, faithfulness) pick up the judge model from
        # the engine unless one was pinned at declaration time.
        if getattr(check, "judge_model", "") is None:
            check.judge_model = judge_model
        outcome = check.evaluate(run, task)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        results.extend(outcome if isinstance(outcome, list) else [outcome])
    return results


async def evaluate_store(
    store,
    tasks: dict[str, Task],
    judge_model: str | None = None,
    max_concurrency: int = 4,
) -> dict[str, int]:
    runs = store.load_runs()
    semaphore = asyncio.Semaphore(max_concurrency)
    counts = {"runs": 0, "checks": 0, "skipped": 0}

    async def _one(run: AgentRun) -> None:
        task = tasks.get(run.task_id)
        if task is None:
            counts["skipped"] += 1
            return
        async with semaphore:
            results = await evaluate_run(run, task, judge_model=judge_model)
        store.save_results(run.run_id, results)
        counts["runs"] += 1
        counts["checks"] += len(results)

    await asyncio.gather(*(_one(r) for r in runs))
    return counts
