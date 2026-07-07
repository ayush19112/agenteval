from agenteval.core.task import task
from agenteval.evaluators.hallucination import Faithfulness
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, OutputNonEmpty, ToolCalled

from ..helpers import with_example
from .mocks import get_stock_price, get_weather


@task(category="tool_use", difficulty="easy", samples=5)
def stock_price():
    return dict(
        input={"prompt": "What is ACME trading at right now?"},
        tools=[get_stock_price],
        reference="ACME is trading at $172.40.",
        checks=[
            ToolCalled("get_stock_price", args={"ticker": "ACME"}),
            Contains("172.40"),
            OutputNonEmpty(),
            Judge("Gives ACME's current price based on the tool output."),
            Faithfulness(),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def london_weather():
    return dict(
        input={"prompt": "What's the weather like in London right now?"},
        tools=[with_example(get_weather, city="London")],
        reference="London is 11°C with light rain.",
        checks=[
            ToolCalled("get_weather", args={"city": "London"}),
            Contains("11"),
            Judge("Reports the current London weather from the tool output."),
        ],
    )
