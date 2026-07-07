from agenteval.core.task import task
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, Regex, ToolCalled

from ..helpers import with_example
from .mocks import lookup_fact


@task(category="reasoning", difficulty="easy", samples=5)
def chain_arithmetic():
    return dict(
        input={"prompt": "Take 7, double it, add 6, then divide by 4. What is the result?"},
        tools=[],
        reference="The result is 5 (7 -> 14 -> 20 -> 5).",
        checks=[
            Regex(r"(?<![\d.])5(?![\d.])"),
            Judge("Arrives at the correct result of 5."),
        ],
    )


@task(category="reasoning", difficulty="medium", samples=5)
def next_in_sequence():
    return dict(
        input={"prompt": "What number comes next in the sequence 2, 6, 12, 20, 30?"},
        tools=[],
        reference="The next number is 42 (differences grow by 2: n(n+1)).",
        checks=[
            Contains("42"),
            Judge("Identifies 42 as the next number."),
        ],
    )


@task(category="reasoning", difficulty="medium", samples=5)
def weekday_reasoning():
    return dict(
        input={
            "prompt": "If March 3, 2025 is a Monday, what day of the week is March 17, 2025?"
        },
        tools=[],
        reference="March 17, 2025 is a Monday (exactly two weeks later).",
        checks=[
            Contains("Monday"),
            Judge("Answers Monday, since March 17 is exactly 14 days after March 3."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def km_to_miles():
    return dict(
        input={"prompt": "Roughly how many miles is 160 kilometers?"},
        tools=[],
        reference="160 km is roughly 99–100 miles.",
        checks=[
            Regex(r"\b(99|100)\b"),
            Judge("Gives approximately 99–100 miles."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def percent_change():
    return dict(
        input={"prompt": "A stock rises from $50 to $65. What is the percent increase?"},
        tools=[],
        reference="That is a 30% increase (15/50).",
        checks=[
            Contains("30"),
            Judge("Gives the correct 30% increase."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def syllogism():
    return dict(
        input={
            "prompt": (
                "All bloops are razzies, and all razzies are lazzies. "
                "Are all bloops lazzies? Answer yes or no, with a one-line reason."
            )
        },
        tools=[],
        reference="Yes — bloops are razzies and razzies are lazzies, so all bloops are lazzies.",
        checks=[
            Regex(r"\byes\b"),
            Judge("Answers yes with valid transitive reasoning."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def odd_one_out():
    return dict(
        input={"prompt": "Which is the odd one out: apple, banana, carrot, mango? Why?"},
        tools=[],
        reference="Carrot — it's a vegetable, while the others are fruits.",
        checks=[
            Contains("carrot"),
            Judge("Picks carrot because it is a vegetable among fruits."),
        ],
    )


@task(category="reasoning", difficulty="medium", samples=5)
def age_puzzle():
    return dict(
        input={
            "prompt": "Sam is twice as old as Ava, and their ages add up to 36. How old is Ava?"
        },
        tools=[],
        reference="Ava is 12 (Sam is 24).",
        checks=[
            Regex(r"\b12\b"),
            Judge("Concludes Ava is 12 years old."),
        ],
    )


@task(category="reasoning", difficulty="medium", samples=5)
def two_hop_capital():
    return dict(
        input={
            "prompt": (
                "What is the population of the capital of Australia? Use the fact lookup tool "
                "(topics: capital_of_australia, canberra_population)."
            )
        },
        tools=[with_example(lookup_fact, topic="capital_of_australia")],
        reference="The capital is Canberra, and its population is about 460,000.",
        checks=[
            ToolCalled("lookup_fact"),
            Contains("Canberra"),
            Judge("Identifies Canberra AND gives its ~460,000 population."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def everest_height():
    return dict(
        input={
            "prompt": (
                "How tall is the tallest mountain on Earth? Use the fact lookup tool "
                "(topic: tallest_mountain)."
            )
        },
        tools=[with_example(lookup_fact, topic="tallest_mountain")],
        reference="Mount Everest is the tallest mountain at 8,849 meters.",
        checks=[
            ToolCalled("lookup_fact"),
            Contains("8,849"),
            Judge("Names Mount Everest at 8,849 meters."),
        ],
    )
