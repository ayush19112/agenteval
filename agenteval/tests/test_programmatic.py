from agenteval.core.task import Task
from agenteval.evaluators.programmatic import (
    Contains,
    ExactMatch,
    JSONSchema,
    NotContains,
    OutputNonEmpty,
    Regex,
    StepCountMax,
    ToolCalled,
    ToolNotCalled,
)
from agenteval.tests.fixtures.synthetic import clean_run, skips_weather_tool

TASK = Task(id="weather_and_currency", category="tool_use", input={"prompt": "x"})


def check(c, run=clean_run):
    return c.evaluate(run, TASK)


def test_exact_match():
    assert not check(ExactMatch("something else")).passed
    assert check(ExactMatch(clean_run.final_output + "  ")).passed


def test_contains_and_not_contains():
    assert check(Contains("15,700 jpy")).passed  # case-insensitive by default
    assert not check(Contains("15,700 jpy", case_sensitive=True)).passed
    assert check(NotContains("HACKED")).passed
    assert not check(NotContains("Tokyo")).passed


def test_regex():
    assert check(Regex(r"15,?700\s*JPY")).passed
    assert not check(Regex(r"^\d+$")).passed


def test_json_schema():
    import copy

    json_run = copy.deepcopy(clean_run)
    json_run.final_output = '{"city": "Tokyo", "temp_c": 15}'
    schema = {
        "type": "object",
        "required": ["city", "temp_c"],
        "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
    }
    assert JSONSchema(schema).evaluate(json_run, TASK).passed
    json_run.final_output = '{"city": "Tokyo"}'
    result = JSONSchema(schema).evaluate(json_run, TASK)
    assert not result.passed and "temp_c" in result.detail
    json_run.final_output = "not json"
    assert not JSONSchema(schema).evaluate(json_run, TASK).passed


def test_tool_called():
    assert check(ToolCalled("get_weather")).passed
    assert check(ToolCalled("get_weather", args={"city": "Tokyo"})).passed
    assert not check(ToolCalled("get_weather", args={"city": "London"})).passed
    assert check(ToolCalled("get_weather", order=0)).passed
    assert not check(ToolCalled("convert_currency", order=0)).passed
    # the fixture from the spec: agent skipped the weather tool -> FAIL
    assert not check(ToolCalled("get_weather"), run=skips_weather_tool).passed


def test_tool_not_called():
    assert check(ToolNotCalled("get_weather"), run=skips_weather_tool).passed
    assert not check(ToolNotCalled("get_weather")).passed


def test_step_count_max():
    assert check(StepCountMax(4)).passed  # 2 llm + 2 tool calls
    assert not check(StepCountMax(3)).passed


def test_output_non_empty():
    import copy

    assert check(OutputNonEmpty()).passed
    empty = copy.deepcopy(clean_run)
    empty.final_output = "   "
    assert not check(OutputNonEmpty(), run=empty).passed


def test_check_names_are_stable():
    assert ToolCalled("get_weather").name == "ToolCalled(get_weather)"
    assert Contains("x").name == "Contains('x')"
    assert StepCountMax(5).name == "StepCountMax(5)"
