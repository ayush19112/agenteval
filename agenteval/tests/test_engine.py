from agenteval.core.task import Task
from agenteval.evaluators.engine import evaluate_run, evaluate_store
from agenteval.evaluators.programmatic import Contains, ToolCalled
from agenteval.tests.fixtures.synthetic import clean_run, skips_weather_tool

TASK = Task(
    id="weather_and_currency",
    category="tool_use",
    input={"prompt": "x"},
    checks=[ToolCalled("get_weather"), ToolCalled("convert_currency"), Contains("15,700")],
)


async def test_evaluate_run_scores_all_checks():
    results = await evaluate_run(clean_run, TASK)
    assert [r.passed for r in results] == [True, True, True]

    results = await evaluate_run(skips_weather_tool, TASK)
    assert [r.passed for r in results] == [False, True, True]


async def test_evaluate_store_writes_results(tmp_path):
    from agenteval.core.store import Store

    store = Store(tmp_path / "t.db")
    store.save_run(clean_run, category="tool_use", model="test")
    store.save_run(skips_weather_tool, category="tool_use", model="test")

    counts = await evaluate_store(store, {"weather_and_currency": TASK})
    assert counts == {"runs": 2, "checks": 6, "skipped": 0}

    loaded = store.load_results()
    assert len(loaded[clean_run.run_id]) == 3
    # re-evaluating is idempotent (results replaced, not duplicated)
    await evaluate_store(store, {"weather_and_currency": TASK})
    assert len(store.load_results()[clean_run.run_id]) == 3
    store.close()
