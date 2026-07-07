from agenteval.core.task import TASK_REGISTRY, load_benchmarks, task


def test_task_decorator_registers():
    @task(category="reasoning", difficulty="easy", samples=2)
    def tiny_example():
        return dict(input={"prompt": "1+1?"}, reference="2")

    t = TASK_REGISTRY.pop("tiny_example")
    assert t.category == "reasoning"
    assert t.samples == 2
    assert tiny_example._agenteval_task is t


def test_load_benchmarks_finds_tasks():
    registry = load_benchmarks()
    assert "weather_and_currency" in registry
    assert registry["weather_and_currency"].category == "tool_use"
