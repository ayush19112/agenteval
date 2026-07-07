from collections import Counter

from agenteval.core.task import CATEGORIES, load_benchmarks


def test_suite_shape():
    registry = load_benchmarks()
    by_category = Counter(t.category for t in registry.values())
    assert len(registry) >= 50, f"suite has only {len(registry)} tasks"
    for category in CATEGORIES:
        assert by_category[category] >= 12, f"too few {category} tasks: {by_category}"


def test_every_task_has_reference_and_checks():
    for t in load_benchmarks().values():
        assert t.reference, f"{t.id} is missing a reference answer"
        assert t.checks, f"{t.id} has no checks"
        assert t.input.get("prompt"), f"{t.id} has no prompt"


def test_tools_are_named_callables():
    for t in load_benchmarks().values():
        for fn in t.tools:
            assert callable(fn) and fn.__name__, f"bad tool in {t.id}"
