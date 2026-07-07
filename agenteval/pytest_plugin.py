"""pytest plugin: @task benchmark functions become pytest tests.

    pytest --pyargs agenteval.benchmarks --eval-model mock/good

Each @task function collects as one test: the agent runs the task
--eval-samples times (default 1 for speed), every check is evaluated, and the
test fails if the mean check score drops below --eval-threshold. Pass
--eval-report scorecard.html to also emit the HTML scorecard.
"""

import asyncio
import os

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("agenteval")
    group.addoption("--eval-model", default=None, help="agent model (default: AGENT_MODEL env)")
    group.addoption("--eval-judge-model", default=None, help="judge model for Judge checks")
    group.addoption("--eval-samples", type=int, default=1, help="samples per task (default 1)")
    group.addoption(
        "--eval-threshold",
        type=float,
        default=0.7,
        help="minimum mean check score to pass (default 0.7)",
    )
    group.addoption("--eval-report", default=None, help="write an HTML scorecard to this path")


def pytest_configure(config):
    config._agenteval_results = []


def pytest_collect_file(file_path, parent):
    """Collect benchmark modules that normal test discovery would skip:
    any .py file under a `benchmarks` directory (they aren't named test_*)."""
    if (
        file_path.suffix == ".py"
        and "benchmarks" in file_path.parts
        and file_path.name != "__init__.py"
        and not file_path.name.startswith("test_")  # leave real tests to pytest itself
    ):
        return pytest.Module.from_parent(parent, path=file_path)


def pytest_pycollect_makeitem(collector, name, obj):
    task = getattr(obj, "_agenteval_task", None)
    if task is not None:
        return EvalItem.from_parent(collector, name=name, task=task)


class EvalFailure(AssertionError):
    pass


class EvalItem(pytest.Item):
    def __init__(self, *, task, **kwargs):
        super().__init__(**kwargs)
        self.task = task

    def _models(self) -> tuple[str, str]:
        from agenteval.core.config import load_config

        cfg = load_config()
        model = self.config.getoption("--eval-model") or cfg.agent_model
        judge = self.config.getoption("--eval-judge-model")
        if judge is None:
            # offline agent + no explicit judge -> offline judge, not a keyed API
            if model.startswith("mock/") and "JUDGE_MODEL" not in os.environ:
                judge = "mock/judge"
            else:
                judge = cfg.judge_model
        return model, judge

    def runtest(self):
        from agenteval.adapters.raw import RawAgent
        from agenteval.core.runner import run_task
        from agenteval.evaluators.engine import evaluate_run

        model, judge = self._models()
        samples = self.config.getoption("--eval-samples")
        threshold = self.config.getoption("--eval-threshold")

        async def _run_all():
            adapter = RawAgent(model)
            outcomes = []
            for i in range(samples):
                run = await run_task(adapter, self.task, i)
                results = await evaluate_run(run, self.task, judge_model=judge)
                outcomes.append((run, results))
            return outcomes

        outcomes = asyncio.run(_run_all())
        self.config._agenteval_results.append((self.task, model, outcomes))

        scores = [
            sum(r.score for r in results) / len(results) if results else 0.0
            for _, results in outcomes
        ]
        mean_score = sum(scores) / len(scores)
        if mean_score < threshold:
            failed = [r for _, results in outcomes for r in results if not r.passed]
            lines = [
                f"task '{self.task.id}' scored {mean_score:.2f} "
                f"(threshold {threshold}, model {model})"
            ]
            lines += [f"  FAIL {r.name}: {r.detail}" for r in failed[:8]]
            raise EvalFailure("\n".join(lines))

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, EvalFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, f"agenteval: {self.task.id} [{self.task.category}]"


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    out = config.getoption("--eval-report", default=None)
    collected = getattr(config, "_agenteval_results", [])
    if not out or not collected:
        return
    from agenteval.core.store import Store
    from agenteval.report.static import render_report

    store = Store(":memory:")
    for task, model, outcomes in collected:
        for run, results in outcomes:
            store.save_run(run, category=task.category, model=model)
            store.save_results(run.run_id, results)
    path = render_report(store, out_path=out)
    store.close()
    print(f"\nagenteval scorecard written to {path}")
