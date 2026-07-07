from agenteval.adapters.raw import RawAgent
from agenteval.core.runner import run_suite
from agenteval.core.store import Store
from agenteval.core.task import Task
from agenteval.evaluators.base import CheckResult


def make_task(task_id="demo", samples=3) -> Task:
    return Task(
        id=task_id, category="tool_use", input={"prompt": "hello"}, tools=[], samples=samples
    )


async def test_run_suite_persists_and_resumes(tmp_path):
    store = Store(tmp_path / "t.db")
    task = make_task(samples=3)
    adapter = RawAgent("mock/good")

    first = await run_suite(adapter, [task], store=store)
    assert len(first) == 3
    assert store.completed_samples("demo", "raw", "mock/good") == {0, 1, 2}

    # resume: everything already stored -> nothing re-runs
    second = await run_suite(adapter, [task], store=store)
    assert second == []

    # raising the sample count only runs the missing samples
    third = await run_suite(adapter, [task], store=store, samples=5)
    assert len(third) == 2
    assert len(store.load_runs()) == 5
    store.close()


async def test_load_runs_filters(tmp_path):
    store = Store(tmp_path / "t.db")
    await run_suite(RawAgent("mock/good"), [make_task()], store=store, samples=1)
    await run_suite(RawAgent("mock/weak"), [make_task()], store=store, samples=1)
    assert len(store.load_runs()) == 2
    assert len(store.load_runs(model="mock/weak")) == 1
    store.close()


def test_results_roundtrip_is_idempotent(tmp_path):
    store = Store(tmp_path / "t.db")
    results = [
        CheckResult(name="Contains(x)", passed=True, score=1.0),
        CheckResult(name="Judge", passed=False, score=0.2, detail="missing JPY amount"),
    ]
    store.save_results("run1", results)
    store.save_results("run1", results)  # re-evaluate: replaces, not duplicates
    loaded = store.load_results()
    assert len(loaded["run1"]) == 2
    assert loaded["run1"][1]["detail"] == "missing JPY amount"
    store.close()
