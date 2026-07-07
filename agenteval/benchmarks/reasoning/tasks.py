from agenteval.core.task import task
from agenteval.evaluators.judge import Judge
from agenteval.evaluators.programmatic import Contains, Regex, ToolCalled

from .mocks import lookup_fact


@task(category="reasoning", difficulty="medium", samples=5)
def multi_hop_film():
    return dict(
        input={
            "prompt": (
                "Which director won Best Picture at the 2020 Oscars, and what is one other "
                "film they directed? Use the fact lookup tool."
            )
        },
        tools=[lookup_fact],
        reference=(
            "Bong Joon-ho won Best Picture for Parasite; he also directed Snowpiercer, "
            "Okja, and Memories of Murder."
        ),
        checks=[
            ToolCalled("lookup_fact"),
            Contains("Bong"),
            Judge("Names the winning director AND at least one other film they directed."),
        ],
    )


@task(category="reasoning", difficulty="easy", samples=5)
def train_speed():
    return dict(
        input={
            "prompt": "A train travels 60 km in 45 minutes. What is its average speed in km/h?"
        },
        tools=[],
        reference="The average speed is 80 km/h.",
        checks=[
            Regex(r"80\s*km"),
            Judge("Gives the correct average speed of 80 km/h."),
        ],
    )


@task(category="reasoning", difficulty="medium", samples=5)
def project_end_date():
    return dict(
        input={
            "prompt": (
                "A project starts on March 3, 2025 and takes exactly 6 weeks. "
                "On what date does it end?"
            )
        },
        tools=[],
        reference="Six weeks after March 3, 2025 is April 14, 2025.",
        checks=[
            Contains("April 14"),
            Judge("Gives the correct end date of April 14, 2025."),
        ],
    )
