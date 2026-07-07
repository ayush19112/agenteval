"""Aggregation: pass@k, cost, and latency over stored runs + results.

Agents are stochastic, so single-run success is misleading. Every task runs
k samples and reports pass@1 (mean), pass@k (any succeeded), pass^k (all
succeeded), and per-task variance to surface flaky tasks.
"""

from collections import defaultdict

from agenteval.core.trace import AgentRun, StepType


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def latency_summary(values: list[float]) -> dict:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def pass_metrics(successes: list[bool]) -> dict:
    n = len(successes)
    p = sum(successes) / n if n else 0.0
    return {
        "n": n,
        "pass_at_1": p,
        "pass_at_k": any(successes) if n else False,
        "pass_all_k": all(successes) if n else False,
        "variance": p * (1 - p),  # Bernoulli variance; >0 means the task is flaky
    }


def run_succeeded(run: AgentRun, results: list[dict]) -> bool:
    return run.error is None and bool(results) and all(r["passed"] for r in results)


def _check_rates(results: list[dict], prefix: str, field: str) -> float | None:
    matching = [r for r in results if r["name"].startswith(prefix)]
    if not matching:
        return None
    return sum(float(r[field]) for r in matching) / len(matching)


def aggregate(runs: list[AgentRun], results: dict[str, list[dict]], meta: dict[str, dict]) -> dict:
    """Build the scorecard: (framework, model) groups -> categories -> tasks.

    `results` and `meta` are keyed by run_id (as returned by the store).
    """
    groups: dict[tuple[str, str], list[AgentRun]] = defaultdict(list)
    for run in runs:
        m = meta.get(run.run_id, {})
        groups[(m.get("framework", run.framework), m.get("model", ""))].append(run)

    scorecard: dict = {"groups": []}
    for (framework, model), group_runs in sorted(groups.items()):
        by_task: dict[str, list[AgentRun]] = defaultdict(list)
        for run in group_runs:
            by_task[run.task_id].append(run)

        tasks = []
        for task_id, task_runs in sorted(by_task.items()):
            task_results = [results.get(r.run_id, []) for r in task_runs]
            successes = [
                run_succeeded(r, res) for r, res in zip(task_runs, task_results, strict=True)
            ]
            flat = [item for res in task_results for item in res]

            check_rates: dict[str, list[float]] = defaultdict(list)
            for res in task_results:
                for r in res:
                    check_rates[r["name"]].append(1.0 if r["passed"] else 0.0)

            failures = [
                {
                    "run_id": r.run_id,
                    "error": r.error,
                    "final_output": r.final_output[:400],
                    "failed_checks": [
                        {"name": c["name"], "detail": c["detail"]}
                        for c in res
                        if not c["passed"]
                    ],
                }
                for r, res, ok in zip(task_runs, task_results, successes, strict=True)
                if not ok
            ]

            tasks.append(
                {
                    "task_id": task_id,
                    "category": meta.get(task_runs[0].run_id, {}).get("category", "unknown"),
                    **pass_metrics(successes),
                    "mean_score": (
                        sum(float(r["score"]) for r in flat) / len(flat) if flat else 0.0
                    ),
                    "mean_cost_usd": sum(r.total_cost_usd for r in task_runs) / len(task_runs),
                    "latency": latency_summary([r.total_latency_ms for r in task_runs]),
                    "llm_ms": sum(
                        s.latency_ms
                        for r in task_runs
                        for s in r.steps
                        if s.type == StepType.LLM_CALL
                    )
                    / len(task_runs),
                    "tool_ms": sum(
                        s.latency_ms
                        for r in task_runs
                        for s in r.steps
                        if s.type == StepType.TOOL_RESULT
                    )
                    / len(task_runs),
                    "checks": [
                        {"name": name, "pass_rate": sum(vals) / len(vals)}
                        for name, vals in check_rates.items()
                    ],
                    "failures": failures,
                }
            )

        categories: dict[str, dict] = {}
        for category in sorted({t["category"] for t in tasks}):
            cat_tasks = [t for t in tasks if t["category"] == category]
            cat_runs = [r for t in cat_tasks for r in by_task[t["task_id"]]]
            categories[category] = {
                "tasks": len(cat_tasks),
                "pass_at_1": sum(t["pass_at_1"] for t in cat_tasks) / len(cat_tasks),
                "pass_at_k": sum(t["pass_at_k"] for t in cat_tasks) / len(cat_tasks),
                "cost_usd": sum(r.total_cost_usd for r in cat_runs),
                "latency": latency_summary([r.total_latency_ms for r in cat_runs]),
            }

        group_flat = [item for r in group_runs for item in results.get(r.run_id, [])]
        scorecard["groups"].append(
            {
                "framework": framework,
                "model": model,
                "n_runs": len(group_runs),
                "n_tasks": len(tasks),
                "n_errors": sum(1 for r in group_runs if r.error),
                # macro average: every task counts equally regardless of sample count
                "pass_at_1": sum(t["pass_at_1"] for t in tasks) / len(tasks) if tasks else 0.0,
                "pass_at_k": sum(t["pass_at_k"] for t in tasks) / len(tasks) if tasks else 0.0,
                "total_cost_usd": sum(r.total_cost_usd for r in group_runs),
                "total_tokens": sum(r.total_tokens for r in group_runs),
                "latency": latency_summary([r.total_latency_ms for r in group_runs]),
                "judge_pass_rate": _check_rates(group_flat, "Judge(", "passed"),
                "faithfulness_mean": _check_rates(group_flat, "Faithfulness", "score"),
                "abstention_rate": _check_rates(group_flat, "Abstained", "passed"),
                "categories": categories,
                "tasks": tasks,
            }
        )
    return scorecard


def scorecard_from_store(store) -> dict:
    return aggregate(store.load_runs(), store.load_results(), store.run_meta())
