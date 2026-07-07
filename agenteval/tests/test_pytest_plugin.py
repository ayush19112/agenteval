"""Phase 11: the pytest plugin collects @task functions and gates on score."""

BENCH = '''
from agenteval.core.task import task
from agenteval.evaluators.programmatic import Contains, ToolCalled


def lookup(city: str) -> str:
    """Look up a city."""
    return f"data for {city}"


lookup.example_args = {"city": "Tokyo"}


@task(category="tool_use", samples=1)
def plugin_demo_task():
    return dict(
        input={"prompt": "What is the data for Tokyo?"},
        tools=[lookup],
        reference="data for Tokyo",
        checks=[ToolCalled("lookup"), Contains("data for Tokyo")],
    )
'''


def test_plugin_collects_and_passes_with_good_agent(pytester):
    pytester.makepyfile(bench=BENCH)
    result = pytester.runpytest("bench.py", "--eval-model", "mock/good", "-v")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*plugin_demo_task*"])


def test_plugin_fails_weak_agent_against_threshold(pytester):
    pytester.makepyfile(bench=BENCH)
    result = pytester.runpytest("bench.py", "--eval-model", "mock/weak")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*scored 0.00*", "*FAIL ToolCalled(lookup)*"])


def test_plugin_discovers_benchmarks_directory(pytester):
    """Modules under a benchmarks/ dir collect even though they aren't test_*.py."""
    bench_dir = pytester.path / "benchmarks"
    bench_dir.mkdir()
    (bench_dir / "my_tasks.py").write_text(BENCH, encoding="utf-8")
    result = pytester.runpytest("benchmarks", "--eval-model", "mock/good")
    result.assert_outcomes(passed=1)


def test_plugin_writes_scorecard_report(pytester):
    pytester.makepyfile(bench=BENCH)
    result = pytester.runpytest(
        "bench.py", "--eval-model", "mock/good", "--eval-report", "scorecard.html"
    )
    result.assert_outcomes(passed=1)
    report = pytester.path / "scorecard.html"
    assert report.exists()
    assert "agenteval scorecard" in report.read_text(encoding="utf-8")
