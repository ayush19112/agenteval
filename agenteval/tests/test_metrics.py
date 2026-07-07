import copy

from agenteval.evaluators.metrics import aggregate, pass_metrics, percentile, run_succeeded
from agenteval.tests.fixtures.synthetic import clean_run, skips_weather_tool


def test_percentile():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0
    assert percentile([], 50) == 0.0


def test_pass_metrics():
    m = pass_metrics([True, True, False, True, False])
    assert m == {
        "n": 5,
        "pass_at_1": 0.6,
        "pass_at_k": True,
        "pass_all_k": False,
        "variance": 0.6 * 0.4,
    }
    stable = pass_metrics([True, True, True])
    assert stable["pass_all_k"] and stable["variance"] == 0.0


def test_run_succeeded_requires_all_checks_and_no_error():
    ok = [{"name": "a", "passed": True, "score": 1.0, "detail": ""}]
    assert run_succeeded(clean_run, ok)
    assert not run_succeeded(clean_run, [])  # no checks -> not a success
    failing = ok + [{"name": "b", "passed": False, "score": 0, "detail": ""}]
    assert not run_succeeded(clean_run, failing)
    errored = copy.deepcopy(clean_run)
    errored.error = "boom"
    assert not run_succeeded(errored, ok)


def test_aggregate_ranks_good_above_weak():
    good = clean_run
    weak = skips_weather_tool
    runs = [good, weak]
    results = {
        good.run_id: [
            {"name": "ToolCalled(get_weather)", "passed": True, "score": 1.0, "detail": ""},
            {"name": "Judge(quality)", "passed": True, "score": 0.9, "detail": ""},
        ],
        weak.run_id: [
            {"name": "ToolCalled(get_weather)", "passed": False, "score": 0.0, "detail": ""},
            {"name": "Judge(quality)", "passed": False, "score": 0.2, "detail": ""},
        ],
    }
    meta = {
        good.run_id: {"framework": "raw", "model": "good", "category": "tool_use"},
        weak.run_id: {"framework": "raw", "model": "weak", "category": "tool_use"},
    }
    card = aggregate(runs, results, meta)
    by_model = {g["model"]: g for g in card["groups"]}
    assert by_model["good"]["pass_at_1"] > by_model["weak"]["pass_at_1"]
    assert by_model["good"]["judge_pass_rate"] == 1.0
    assert by_model["weak"]["judge_pass_rate"] == 0.0
    assert by_model["good"]["categories"]["tool_use"]["tasks"] == 1
    task = by_model["weak"]["tasks"][0]
    assert task["failures"] and task["failures"][0]["failed_checks"]
