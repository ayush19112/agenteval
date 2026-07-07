from agenteval.calibration.calibrate import (
    KAPPA_THRESHOLD,
    calibrate,
    cohens_kappa,
    load_golden_set,
)
from agenteval.core.task import Task
from agenteval.evaluators.judge import Judge, _extract_json, _heuristic_grade
from agenteval.tests.fixtures.synthetic import clean_run, skips_weather_tool

TASK = Task(
    id="weather_and_currency",
    category="tool_use",
    input={"prompt": "What's the weather in Tokyo, and 100 USD in JPY?"},
    reference="About 15°C and partly cloudy in Tokyo; 100 USD is roughly 15,700 JPY.",
)


def test_extract_json_handles_fences():
    assert _extract_json('```json\n{"passed": true, "score": 1.0}\n```')["passed"] is True


def test_heuristic_grade_directions():
    ref = "ACME is trading at $172.40."
    assert _heuristic_grade("ACME is trading at $172.40 today.", ref)[0]
    assert not _heuristic_grade("ACME is at $89.20.", ref)[0]
    assert not _heuristic_grade("answer", None)[0]  # no reference -> can't grade


async def test_judge_check_with_mock_model():
    judge = Judge("Gives both weather and a JPY amount.", model="mock/judge")
    good = await judge.evaluate(clean_run, TASK)
    assert good.passed and good.score > 0.6

    import copy

    bad = copy.deepcopy(skips_weather_tool)
    bad.final_output = "I cannot say."
    result = await judge.evaluate(bad, TASK)
    assert not result.passed


def test_cohens_kappa_bounds():
    assert cohens_kappa([True, False] * 5, [True, False] * 5) == 1.0
    assert cohens_kappa([True, False] * 5, [False, True] * 5) < 0


def test_golden_set_is_wellformed():
    entries = load_golden_set()
    assert len(entries) >= 30
    labels = {e["human_passed"] for e in entries}
    assert labels == {True, False}  # both classes represented
    assert len({e["id"] for e in entries}) == len(entries)  # unique ids


async def test_mock_judge_clears_kappa_threshold():
    """The calibration gate the spec requires — offline via the mock judge."""
    report = await calibrate("mock/judge")
    assert report["kappa"] >= KAPPA_THRESHOLD, report["disagreements"]
