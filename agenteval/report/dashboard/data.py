"""Builds the JSON payload the React dashboard consumes.

Shape mirrors the dashboard's data contract: agents (per framework+model
scorecards), per-agent task rows with a representative trace, a pass@k trend,
and run/judge metadata.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agenteval.core.trace import AgentRun, StepType
from agenteval.evaluators.metrics import aggregate, run_succeeded


def _agent_id(framework: str, model: str) -> str:
    return f"{framework}|{model}"


def _short(framework: str, model: str) -> str:
    return f"{framework}·{model.split('/')[-1]}"


def _pct(x: float | None) -> int | None:
    return None if x is None else round(x * 100)


def _trace_steps(run: AgentRun) -> list[dict]:
    steps = []
    for s in run.steps:
        if s.type == StepType.LLM_CALL:
            steps.append(
                {
                    "type": "llm_call",
                    "label": s.model or "llm",
                    "ms": round(s.latency_ms),
                    "note": (s.content or "")[:80] or None,
                    "tokensIn": s.tokens_in,
                    "tokensOut": s.tokens_out,
                }
            )
        elif s.type == StepType.TOOL_CALL:
            steps.append(
                {
                    "type": "tool_call",
                    "label": s.tool_name,
                    "ms": round(s.latency_ms),
                    "note": json.dumps(s.tool_args or {})[:80],
                }
            )
        else:
            steps.append(
                {
                    "type": "tool_result",
                    "label": s.tool_name,
                    "ms": round(s.latency_ms),
                    "note": (s.tool_output or "")[:80],
                }
            )
    return steps


def _task_checks(sample_results: list[list[dict]]) -> list[dict]:
    """Aggregate check results across a task's samples into one row per check."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for results in sample_results:
        for r in results:
            by_name[r["name"]].append(r)
    rows = []
    for name, items in by_name.items():
        rate = sum(1 for r in items if r["passed"]) / len(items)
        failing = next((r for r in items if not r["passed"]), None)
        detail = (failing or items[0]).get("detail", "")
        if 0 < rate < 1:
            passed_n = sum(1 for r in items if r["passed"])
            detail = f"{passed_n}/{len(items)} samples passed. {detail}".strip()
        rows.append(
            {
                "name": name,
                "passed": rate >= 0.5,
                "score": round(sum(float(r["score"]) for r in items) / len(items), 2),
                "detail": detail,
            }
        )
    return rows


def dashboard_payload(store, calibration_path: str | None = None) -> dict:
    runs = store.load_runs()
    results = store.load_results()
    meta = store.run_meta()
    card = aggregate(runs, results, meta)

    agents, tasks_by_agent, trend_by_agent = [], {}, {}
    for group in card["groups"]:
        framework, model = group["framework"], group["model"]
        agent_id = _agent_id(framework, model)
        agents.append(
            {
                "id": agent_id,
                "label": f"{framework} · {model}",
                "short": _short(framework, model),
                "passK": _pct(group["pass_at_k"]),
                "pass1": _pct(group["pass_at_1"]),
                "faith": _pct(group["faithfulness_mean"]),
                "costRun": (
                    group["total_cost_usd"] / group["n_runs"] if group["n_runs"] else 0.0
                ),
                "p50": round(group["latency"]["p50"]),
                "p95": round(group["latency"]["p95"]),
                "cats": {
                    cat: _pct(c["pass_at_k"]) for cat, c in group["categories"].items()
                },
            }
        )

        group_runs = [
            r
            for r in runs
            if meta.get(r.run_id, {}).get("framework") == framework
            and meta.get(r.run_id, {}).get("model") == model
        ]
        by_task: dict[str, list[AgentRun]] = defaultdict(list)
        for r in group_runs:
            by_task[r.task_id].append(r)

        task_rows = []
        for t in group["tasks"]:
            task_runs = by_task[t["task_id"]]
            sample_results = [results.get(r.run_id, []) for r in task_runs]
            representative = next(
                (
                    r
                    for r, res in zip(task_runs, sample_results, strict=True)
                    if not run_succeeded(r, res)
                ),
                task_runs[0] if task_runs else None,
            )
            judge_scores = [
                float(r["score"])
                for res in sample_results
                for r in res
                if r["name"].startswith("Judge(")
            ]
            task_rows.append(
                {
                    "id": t["task_id"],
                    "category": t["category"],
                    "samples": t["n"],
                    "passed": round(t["pass_at_1"] * t["n"]),
                    "ok": t["pass_at_1"] >= 0.5,
                    "cost": t["mean_cost_usd"],
                    "p50": round(t["latency"]["p50"]),
                    "judge": round(
                        sum(judge_scores) / len(judge_scores) if judge_scores else t["mean_score"],
                        2,
                    ),
                    "prompt": representative.input.get("prompt") if representative else None,
                    "output": representative.final_output[:300] if representative else None,
                    "checks": _task_checks(sample_results),
                    "trace": _trace_steps(representative) if representative else [],
                }
            )
        tasks_by_agent[agent_id] = task_rows

        by_date: dict[str, list[bool]] = defaultdict(list)
        for r, res in ((r, results.get(r.run_id, [])) for r in group_runs):
            created = meta.get(r.run_id, {}).get("created_at")
            day = datetime.fromtimestamp(created).strftime("%b %d") if created else "run"
            by_date[day].append(run_succeeded(r, res))
        trend_by_agent[agent_id] = [
            {"run": day, "passK": round(sum(oks) / len(oks) * 100)}
            for day, oks in sorted(by_date.items())
        ]

    calibration = {}
    cal_file = Path(calibration_path or "judge_calibration.json")
    if cal_file.exists():
        calibration = json.loads(cal_file.read_text(encoding="utf-8"))

    dates = [m["created_at"] for m in meta.values() if m.get("created_at")]
    run_id = (
        "run_" + datetime.fromtimestamp(max(dates)).strftime("%Y-%m-%d_%H:%M") if dates else "run"
    )
    return {
        "run": {
            "id": run_id,
            "judge": calibration.get("model"),
            "kappa": calibration.get("kappa"),
        },
        "agents": agents,
        "tasks": tasks_by_agent,
        "trend": trend_by_agent,
    }
