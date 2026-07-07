from agenteval.core.store import Store
from agenteval.core.task import Task
from agenteval.evaluators.engine import evaluate_store
from agenteval.evaluators.programmatic import Contains, ToolCalled
from agenteval.report.dashboard.data import dashboard_payload
from agenteval.tests.fixtures.synthetic import clean_run, skips_weather_tool

TASK = Task(
    id="weather_and_currency",
    category="tool_use",
    input={"prompt": "x"},
    checks=[ToolCalled("get_weather"), Contains("15,700")],
)


async def test_dashboard_payload_shape(tmp_path):
    store = Store(tmp_path / "t.db")
    store.save_run(clean_run, category="tool_use", model="mock/good")
    store.save_run(skips_weather_tool, category="tool_use", model="mock/weak")
    await evaluate_store(store, {"weather_and_currency": TASK})

    payload = dashboard_payload(store, calibration_path=str(tmp_path / "missing.json"))

    assert {a["id"] for a in payload["agents"]} == {"raw|mock/good", "raw|mock/weak"}
    good = next(a for a in payload["agents"] if a["id"] == "raw|mock/good")
    weak = next(a for a in payload["agents"] if a["id"] == "raw|mock/weak")
    assert good["passK"] == 100 and weak["passK"] == 0
    assert good["cats"]["tool_use"] == 100

    weak_tasks = payload["tasks"]["raw|mock/weak"]
    assert weak_tasks[0]["ok"] is False
    assert any(not c["passed"] for c in weak_tasks[0]["checks"])
    # representative trace comes from a failing run and keeps step order
    assert weak_tasks[0]["trace"][0]["type"] == "llm_call"
    assert payload["trend"]["raw|mock/good"][0]["passK"] == 100
    assert payload["run"]["kappa"] is None  # no calibration file -> badge hidden
    store.close()
