from agenteval.core.task import task
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, StepCountMax, ToolCalled

from ..helpers import with_example
from ..tool_use.mocks import (
    calculator,
    get_contact,
    get_distance,
    get_stock_price,
    get_weather,
    send_email,
)
from .mocks import (
    add_calendar,
    add_to_cart,
    check_inventory,
    checkout,
    get_balance,
    get_calendar,
    get_flight_price,
    get_order_status,
    list_files,
    place_order,
    read_file,
    search_products,
    transfer,
)


@task(category="long_horizon", difficulty="medium", samples=5)
def shopping_budget():
    return dict(
        input={"prompt": "Buy 3 units of the cheapest widget in the catalog and place the order."},
        tools=[search_products, add_to_cart, checkout],
        reference=(
            "The cheapest widget is widget-mini ($10); 3 were added to the cart and the order "
            "was placed with confirmation CHK-9."
        ),
        checks=[
            ToolCalled("search_products"),
            ToolCalled("add_to_cart", args={"item": "widget-mini"}),
            ToolCalled("checkout"),
            Contains("CHK-9"),
            StepCountMax(12),
            Judge("Buys 3 of the cheapest widget (widget-mini) and confirms the order."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def bank_transfer():
    return dict(
        input={
            "prompt": (
                "Check my checking balance, then move $200 from checking to savings and tell "
                "me the new balances."
            )
        },
        tools=[get_balance, transfer],
        reference=(
            "Checking had $1,250.00; after moving $200, checking is $1,050.00 and savings is "
            "$8,500.00."
        ),
        checks=[
            ToolCalled("get_balance"),
            ToolCalled("transfer", args={"amount": 200}),
            Contains("1,050"),
            StepCountMax(10),
            Judge("Reports the new balances after the $200 transfer."),
        ],
    )


@task(category="long_horizon", difficulty="hard", samples=5)
def two_city_trip():
    return dict(
        input={
            "prompt": (
                "I'm flying from London to Cairo. What does the flight cost, how far is it, "
                "and what's the weather there?"
            )
        },
        tools=[
            with_example(get_flight_price, from_city="London", to_city="Cairo"),
            with_example(get_distance, city_a="London", city_b="Cairo"),
            with_example(get_weather, city="Cairo"),
        ],
        reference=(
            "The London–Cairo flight costs 320 USD, the distance is about 3,510 km, and Cairo "
            "is 31°C with clear skies."
        ),
        checks=[
            ToolCalled("get_flight_price"),
            ToolCalled("get_distance"),
            ToolCalled("get_weather", args={"city": "Cairo"}),
            Contains("320"),
            StepCountMax(10),
            Judge("Gives the price (320 USD), distance (~3,510 km), and Cairo weather."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def meeting_schedule():
    return dict(
        input={
            "prompt": (
                "Check my Friday calendar and book a 1:1 with Bob at a free time in the "
                "afternoon."
            )
        },
        tools=[get_calendar, add_calendar],
        reference=(
            "Friday has a 9:00 standup and a 14:00 design review, so the 1:1 with Bob was "
            "booked at a free afternoon slot such as 15:00."
        ),
        checks=[
            ToolCalled("get_calendar"),
            ToolCalled("add_calendar"),
            StepCountMax(10),
            Judge("Books the 1:1 at a time that does not clash with 9:00 or 14:00."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def stock_portfolio():
    return dict(
        input={
            "prompt": (
                "What are 10 shares of ACME worth right now? Get the price, then use the "
                "calculator."
            )
        },
        tools=[get_stock_price, with_example(calculator, expression="172.40 * 10")],
        reference="ACME is at $172.40, so 10 shares are worth $1,724.",
        checks=[
            ToolCalled("get_stock_price", args={"ticker": "ACME"}),
            ToolCalled("calculator"),
            Contains("1,724"),
            StepCountMax(10),
            Judge("Computes ~$1,724 for 10 ACME shares."),
        ],
    )


@task(category="long_horizon", difficulty="hard", samples=5)
def inventory_restock():
    return dict(
        input={
            "prompt": (
                "If 'gadget' is out of stock, order 20 units and confirm the order status."
            )
        },
        tools=[
            with_example(check_inventory, item="gadget"),
            with_example(place_order, item="gadget", quantity=20),
            get_order_status,
        ],
        reference=(
            "Gadgets were out of stock (0 units), so 20 were ordered (ORD-7) and the order "
            "status is confirmed."
        ),
        checks=[
            ToolCalled("check_inventory", args={"item": "gadget"}),
            ToolCalled("place_order"),
            ToolCalled("get_order_status"),
            StepCountMax(12),
            Judge("Orders 20 gadgets because stock was 0, and reports the confirmed status."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def email_report():
    return dict(
        input={
            "prompt": (
                "Read report.txt and email a one-line status summary to alice@example.com."
            )
        },
        tools=[
            with_example(read_file, name="report.txt"),
            with_example(
                send_email,
                to="alice@example.com",
                subject="Status summary",
                body="The launch is on track with three risks.",
            ),
        ],
        reference=(
            "report.txt says the launch is on track with three risks, and that summary was "
            "emailed to alice@example.com."
        ),
        checks=[
            ToolCalled("read_file"),
            ToolCalled("send_email", args={"to": "alice@example.com"}),
            StepCountMax(10),
            Judge("Emails Alice a summary that reflects the report contents."),
        ],
    )


@task(category="long_horizon", difficulty="hard", samples=5)
def distance_and_time():
    return dict(
        input={
            "prompt": (
                "How far is London from Tokyo, and how long would that take at 800 km/h? "
                "Use the tools."
            )
        },
        tools=[get_distance, with_example(calculator, expression="9560 / 800")],
        reference="London to Tokyo is about 9,560 km — roughly 12 hours at 800 km/h.",
        checks=[
            ToolCalled("get_distance"),
            ToolCalled("calculator"),
            Contains("11.9"),
            StepCountMax(10),
            Judge("Gives ~9,560 km and roughly 12 hours of flight time."),
        ],
    )


@task(category="long_horizon", difficulty="hard", samples=5)
def project_status_rollup():
    return dict(
        input={
            "prompt": (
                "Find the project files, read the report, and email its status to "
                "alice@example.com."
            )
        },
        tools=[
            list_files,
            with_example(read_file, name="report.txt"),
            with_example(
                send_email,
                to="alice@example.com",
                subject="Project status",
                body="The launch is on track with three risks.",
            ),
        ],
        reference=(
            "The project files include report.txt, which says the launch is on track with "
            "three risks; that status was emailed to alice@example.com."
        ),
        checks=[
            ToolCalled("list_files"),
            ToolCalled("read_file"),
            ToolCalled("send_email"),
            StepCountMax(12),
            Judge("Reads the report and emails its status to Alice."),
        ],
    )


@task(category="long_horizon", difficulty="medium", samples=5)
def refund_flow():
    return dict(
        input={
            "prompt": (
                "Check the status of order ORD-7; if it's confirmed, look up Bob's email and "
                "send him a confirmation note."
            )
        },
        tools=[
            get_order_status,
            get_contact,
            with_example(
                send_email,
                to="bob@example.com",
                subject="Order confirmed",
                body="Order ORD-7 is confirmed.",
            ),
        ],
        reference=(
            "Order ORD-7 is confirmed; Bob's address is bob@example.com and a confirmation "
            "note was sent to him."
        ),
        checks=[
            ToolCalled("get_order_status", args={"order_id": "ORD-7"}),
            ToolCalled("get_contact"),
            ToolCalled("send_email"),
            StepCountMax(12),
            Judge("Confirms ORD-7 and emails Bob about it."),
        ],
    )
