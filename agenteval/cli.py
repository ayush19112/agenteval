"""CLI: `agenteval run` / `agenteval evaluate` / `agenteval report`.

run      — execute agents on benchmark tasks, persist AgentRuns to SQLite
evaluate — score stored runs (repeatable, never re-runs the agent)
report   — render the HTML scorecard from stored runs + results
"""

import argparse
import asyncio


def _build_adapter(framework: str, model: str):
    if framework == "raw":
        from agenteval.adapters.raw import RawAgent

        return RawAgent(model)
    if framework == "langgraph":
        from agenteval.adapters.langgraph import LangGraphAgent

        return LangGraphAgent(model)
    if framework == "crewai":
        from agenteval.adapters.crewai import CrewAIAgent

        return CrewAIAgent(model)
    raise SystemExit(f"Unknown framework '{framework}'. Available: raw, langgraph, crewai.")


def _select_tasks(args) -> list:
    from agenteval.core.task import load_benchmarks

    registry = load_benchmarks()
    if getattr(args, "task", None):
        if args.task not in registry:
            raise SystemExit(f"Unknown task '{args.task}'. Known: {', '.join(sorted(registry))}")
        return [registry[args.task]]
    tasks = list(registry.values())
    if getattr(args, "category", None):
        tasks = [t for t in tasks if t.category == args.category]
    return tasks


def cmd_run(args) -> None:
    from agenteval.core.config import load_config
    from agenteval.core.runner import run_suite
    from agenteval.core.store import Store

    cfg = load_config()
    model = args.model or cfg.agent_model
    tasks = _select_tasks(args)
    adapter = _build_adapter(args.framework, model)
    store = Store(args.db or cfg.db_path)

    runs = asyncio.run(
        run_suite(
            adapter,
            tasks,
            store=store,
            samples=args.samples,
            max_concurrency=cfg.max_concurrency,
            resume=not args.no_resume,
        )
    )
    errors = [r for r in runs if r.error]
    print(
        f"ran {len(runs)} sample(s) across {len(tasks)} task(s) "
        f"[framework={args.framework} model={model}] -> {store.path}"
    )
    if errors:
        print(f"  {len(errors)} run(s) errored, e.g.: {errors[0].error}")
    if args.print_trace:
        for run in runs:
            print(run.model_dump_json(indent=2))
    store.close()


def cmd_evaluate(args) -> None:
    from agenteval.core.config import load_config
    from agenteval.core.store import Store
    from agenteval.core.task import load_benchmarks
    from agenteval.evaluators.engine import evaluate_store

    cfg = load_config()
    store = Store(args.db or cfg.db_path)
    tasks = load_benchmarks()
    judge_model = args.judge_model or cfg.judge_model
    counts = asyncio.run(evaluate_store(store, tasks, judge_model=judge_model))
    print(
        f"evaluated {counts['runs']} run(s): {counts['checks']} check result(s) "
        f"written to {store.path}"
    )
    store.close()


def cmd_report(args) -> None:
    from agenteval.core.config import load_config
    from agenteval.core.store import Store
    from agenteval.report.static import render_report

    cfg = load_config()
    store = Store(args.db or cfg.db_path)
    path = render_report(store, out_path=args.out)
    print(f"scorecard written to {path}")
    store.close()


def cmd_dashboard(args) -> None:
    from agenteval.core.config import load_config
    from agenteval.report.dashboard.server import serve

    cfg = load_config()
    serve(args.db or cfg.db_path, port=args.port)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="agenteval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run agents on benchmark tasks and persist traces")
    p_run.add_argument("--task", help="run a single task by id")
    p_run.add_argument("--category", help="filter tasks by category")
    p_run.add_argument("--framework", default="raw")
    p_run.add_argument("--model", help="LiteLLM model string, or mock/good | mock/weak")
    p_run.add_argument("--samples", type=int, help="override samples per task")
    p_run.add_argument("--db", help="SQLite path (default: agenteval.db)")
    p_run.add_argument("--no-resume", action="store_true", help="re-run already-stored samples")
    p_run.add_argument("--print-trace", action="store_true", help="print full AgentRun JSON")
    p_run.set_defaults(fn=cmd_run)

    p_eval = sub.add_parser("evaluate", help="score stored runs (no agent execution)")
    p_eval.add_argument("--db")
    p_eval.add_argument("--judge-model", help="LiteLLM model string, or mock/judge")
    p_eval.set_defaults(fn=cmd_evaluate)

    p_report = sub.add_parser("report", help="render the HTML scorecard")
    p_report.add_argument("--db")
    p_report.add_argument("--out", default="report.html")
    p_report.set_defaults(fn=cmd_report)

    p_dash = sub.add_parser("dashboard", help="serve the interactive React dashboard")
    p_dash.add_argument("--db")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.set_defaults(fn=cmd_dashboard)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
