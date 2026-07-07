from agenteval.core.task import task
from agenteval.evaluators.hallucination import Faithfulness
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, ToolCalled

from .mocks import convert_currency, get_weather


@task(category="tool_use", difficulty="medium", samples=5)
def weather_and_currency():
    return dict(
        input={"prompt": "What's the weather in Tokyo, and how much is 100 USD in JPY?"},
        tools=[get_weather, convert_currency],
        reference="About 15°C and partly cloudy in Tokyo; 100 USD is roughly 15,700 JPY.",
        checks=[
            ToolCalled("get_weather", args={"city": "Tokyo"}),
            ToolCalled("convert_currency"),
            Contains("15,700"),
            Judge("Does the answer give both the Tokyo weather and a JPY amount?"),
            Faithfulness(),
        ],
    )
