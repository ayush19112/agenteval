from agenteval.core.task import task
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, StepCountMax, ToolCalled

from ..helpers import with_example
from ..tool_use.mocks import convert_currency, get_weather
from .mocks import (
    check_inventory,
    get_flight_price,
    get_order_status,
    list_files,
    place_order,
    read_file,
)


@task(category="long_horizon", difficulty="hard", samples=5)
def trip_budget():
    return dict(
        input={
            "prompt": (
                "I'm flying from London to Tokyo. Find the flight price, convert it to JPY, "
                "and tell me the weather at my destination."
            )
        },
        tools=[
            get_flight_price,
            with_example(convert_currency, amount=850, from_currency="USD", to_currency="JPY"),
            with_example(get_weather, city="Tokyo"),
        ],
        reference=(
            "The flight costs about 850 USD, roughly 133,450 JPY, and Tokyo is currently "
            "15°C and partly cloudy."
        ),
        checks=[
            ToolCalled("get_flight_price"),
            ToolCalled("convert_currency"),
            ToolCalled("get_weather", args={"city": "Tokyo"}),
            Contains("133,450"),
            StepCountMax(10),
            Judge("Gives the flight price, its JPY conversion, and the Tokyo weather."),
        ],
    )


@task(category="long_horizon", difficulty="hard", samples=5)
def file_word_count():
    return dict(
        input={
            "prompt": (
                "Using the file tools, read every .txt file in the project and report the "
                "total word count across them."
            )
        },
        tools=[list_files, read_file],
        reference="notes.txt has 6 words and report.txt has 8 words, so the total is 14 words.",
        checks=[
            ToolCalled("list_files"),
            ToolCalled("read_file"),
            StepCountMax(14),
            Judge("Reports the correct total word count of 14 across the two .txt files."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def widget_order():
    return dict(
        input={
            "prompt": (
                "Order 3 units of 'widget' if they're in stock, then tell me the order status."
            )
        },
        tools=[check_inventory, place_order, get_order_status],
        reference=(
            "Widgets were in stock (42 units), order ORD-7 was placed for 3 units, and its "
            "status is confirmed."
        ),
        checks=[
            ToolCalled("check_inventory", args={"item": "widget"}),
            ToolCalled("place_order"),
            ToolCalled("get_order_status"),
            StepCountMax(12),
            Judge("Confirms the order was placed and reports its status."),
        ],
    )
