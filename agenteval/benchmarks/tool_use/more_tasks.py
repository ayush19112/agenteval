from agenteval.core.task import task
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, ToolCalled

from ..helpers import with_example
from .mocks import (
    calculator,
    convert_currency,
    get_contact,
    get_definition,
    get_distance,
    get_population,
    get_stock_price,
    get_weather,
    send_email,
    translate,
)


@task(category="tool_use", difficulty="easy", samples=5)
def cairo_weather():
    return dict(
        input={"prompt": "What's the weather in Cairo right now?"},
        tools=[with_example(get_weather, city="Cairo")],
        reference="Cairo is 31°C with clear skies.",
        checks=[
            ToolCalled("get_weather", args={"city": "Cairo"}),
            Contains("31"),
            Judge("Reports the current Cairo weather from the tool output."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def eur_conversion():
    return dict(
        input={"prompt": "How much is 250 EUR in USD?"},
        tools=[with_example(convert_currency, amount=250, from_currency="EUR", to_currency="USD")],
        reference="250 EUR is about 270 USD.",
        checks=[
            ToolCalled("convert_currency"),
            Contains("270"),
            Judge("Gives the correct USD amount for 250 EUR."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def globex_price():
    return dict(
        input={"prompt": "What is GLOBEX trading at?"},
        tools=[with_example(get_stock_price, ticker="GLOBEX")],
        reference="GLOBEX is trading at $54.10.",
        checks=[
            ToolCalled("get_stock_price", args={"ticker": "GLOBEX"}),
            Contains("54.10"),
            Judge("Gives GLOBEX's current price from the tool output."),
        ],
    )


@task(category="tool_use", difficulty="medium", samples=5)
def tip_calculator():
    return dict(
        input={"prompt": "Use the calculator to work out a 17% tip on an $84.50 bill."},
        tools=[calculator],
        reference="A 17% tip on $84.50 is about $14.37.",
        checks=[
            ToolCalled("calculator"),
            Contains("14.3"),
            Judge("Gives roughly $14.37 as the tip, computed with the calculator."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def translate_hello():
    return dict(
        input={"prompt": "How do you say 'hello' in Spanish?"},
        tools=[translate],
        reference="'hello' in Spanish is 'hola'.",
        checks=[
            ToolCalled("translate"),
            Contains("hola"),
            Judge("Gives the Spanish translation 'hola'."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def define_ephemeral():
    return dict(
        input={"prompt": "What does the word 'ephemeral' mean? Use the dictionary tool."},
        tools=[get_definition],
        reference="Ephemeral means lasting for a very short time.",
        checks=[
            ToolCalled("get_definition", args={"word": "ephemeral"}),
            Contains("short time"),
            Judge("Defines ephemeral as lasting a very short time."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def tokyo_population():
    return dict(
        input={"prompt": "What's the population of Tokyo?"},
        tools=[get_population],
        reference="Tokyo has about 14 million people (city proper).",
        checks=[
            ToolCalled("get_population", args={"city": "Tokyo"}),
            Contains("14 million"),
            Judge("Gives Tokyo's population of roughly 14 million."),
        ],
    )


@task(category="tool_use", difficulty="medium", samples=5)
def email_alice():
    return dict(
        input={
            "prompt": "Send an email to alice@example.com letting her know the report is ready."
        },
        tools=[send_email],
        reference="An email was sent to alice@example.com saying the report is ready.",
        checks=[
            ToolCalled("send_email", args={"to": "alice@example.com"}),
            Contains("alice@example.com"),
            Judge("Confirms an email to alice@example.com about the report being ready."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def contact_lookup():
    return dict(
        input={"prompt": "What's Bob's email address? Check the contact list."},
        tools=[get_contact],
        reference="Bob's email address is bob@example.com.",
        checks=[
            ToolCalled("get_contact"),
            Contains("bob@example.com"),
            Judge("Gives Bob's address bob@example.com from the contacts tool."),
        ],
    )


@task(category="tool_use", difficulty="easy", samples=5)
def london_tokyo_distance():
    return dict(
        input={"prompt": "How far is London from Tokyo?"},
        tools=[get_distance],
        reference="London to Tokyo is about 9,560 km.",
        checks=[
            ToolCalled("get_distance"),
            Contains("9,560"),
            Judge("Gives the London–Tokyo distance of roughly 9,560 km."),
        ],
    )
