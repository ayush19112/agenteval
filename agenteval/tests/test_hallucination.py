from agenteval.core.task import Task
from agenteval.evaluators.hallucination import Abstained, Faithfulness
from agenteval.tests.fixtures.synthetic import (
    abstains_correctly,
    clean_run,
    fabricates_claim,
    skips_weather_tool,
)

WEATHER_TASK = Task(
    id="weather_and_currency",
    category="tool_use",
    input={"prompt": "x"},
    reference="About 15°C and partly cloudy in Tokyo; 100 USD is roughly 15,700 JPY.",
)
STOCK_TASK = Task(
    id="stock_price",
    category="tool_use",
    input={"prompt": "x"},
    reference="ACME is trading at $172.40.",
)
UNANSWERABLE_TASK = Task(id="unanswerable_entity", category="adversarial", input={"prompt": "x"})


async def test_faithful_run_passes():
    result = await Faithfulness(model="mock/judge").evaluate(clean_run, WEATHER_TASK)
    assert result.passed and result.score == 1.0


async def test_fabricated_weather_claim_is_flagged():
    # spec Appendix B: the fabricated "sunny in Tokyo" claim must be caught
    result = await Faithfulness(model="mock/judge").evaluate(skips_weather_tool, WEATHER_TASK)
    assert not result.passed
    assert result.score < 0.7
    assert "sunny" in result.detail.lower()


async def test_fully_fabricated_answer_scores_zero():
    result = await Faithfulness(model="mock/judge").evaluate(fabricates_claim, STOCK_TASK)
    assert not result.passed and result.score == 0.0


def test_abstained_passes_correct_refusal():
    result = Abstained().evaluate(abstains_correctly, UNANSWERABLE_TASK)
    assert result.passed


def test_abstained_fails_confident_fabrication():
    result = Abstained().evaluate(fabricates_claim, UNANSWERABLE_TASK)
    assert not result.passed
    assert result.detail == "answered confidently"
