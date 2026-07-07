"""Hand-written AgentRun traces — the bulk of the test suite runs on these,
with zero LLM calls. Each fixture models one behavior the platform must catch."""

from agenteval.core.trace import AgentRun, Step, StepType

# A clean, successful run: both tools called correctly, answer grounded in evidence.
clean_run = AgentRun(
    task_id="weather_and_currency",
    run_id="weather_and_currency:0:clean",
    framework="raw",
    input={"prompt": "What's the weather in Tokyo, and 100 USD in JPY?"},
    steps=[
        Step(type=StepType.LLM_CALL, model="test", tokens_in=90, tokens_out=10, latency_ms=350),
        Step(type=StepType.TOOL_CALL, tool_name="get_weather", tool_args={"city": "Tokyo"}),
        Step(type=StepType.TOOL_RESULT, tool_name="get_weather", tool_output="15°C, partly cloudy"),
        Step(
            type=StepType.TOOL_CALL,
            tool_name="convert_currency",
            tool_args={"amount": 100, "from_currency": "USD", "to_currency": "JPY"},
        ),
        Step(
            type=StepType.TOOL_RESULT,
            tool_name="convert_currency",
            tool_output="100 USD = 15,700 JPY",
        ),
        Step(type=StepType.LLM_CALL, model="test", tokens_in=200, tokens_out=40, latency_ms=420),
    ],
    final_output="It's 15°C and partly cloudy in Tokyo, and 100 USD is about 15,700 JPY.",
    total_tokens=340,
    total_cost_usd=0.0003,
    total_latency_ms=800,
)

# An agent that answered without calling the required weather tool:
# the weather claim is fabricated (spec Appendix B).
skips_weather_tool = AgentRun(
    task_id="weather_and_currency",
    run_id="weather_and_currency:0:skips",
    framework="raw",
    input={"prompt": "What's the weather in Tokyo, and 100 USD in JPY?"},
    steps=[
        Step(type=StepType.LLM_CALL, model="test", tokens_in=40, tokens_out=20, latency_ms=300),
        Step(type=StepType.TOOL_CALL, tool_name="convert_currency", tool_args={"amount": 100}),
        Step(
            type=StepType.TOOL_RESULT,
            tool_name="convert_currency",
            tool_output="100 USD = 15,700 JPY",
        ),
        Step(type=StepType.LLM_CALL, model="test", tokens_in=90, tokens_out=25, latency_ms=310),
    ],
    final_output="It's sunny in Tokyo, and 100 USD is about 15,700 JPY.",  # weather fabricated
    total_tokens=175,
    total_latency_ms=610,
)

# A confident answer with a claim not supported by any tool output.
fabricates_claim = AgentRun(
    task_id="stock_price",
    run_id="stock_price:0:fabricates",
    framework="raw",
    input={"prompt": "What is ACME trading at?"},
    steps=[
        Step(type=StepType.LLM_CALL, model="test", tokens_in=50, tokens_out=30, latency_ms=280),
    ],
    final_output="ACME is trading at $999.99 and just announced a merger with Globex.",
    total_tokens=80,
    total_latency_ms=280,
)

# Correct abstention on an unanswerable task.
abstains_correctly = AgentRun(
    task_id="unanswerable_entity",
    run_id="unanswerable_entity:0:abstains",
    framework="raw",
    input={"prompt": "What is the population of Zarqonia in 2024?"},
    steps=[
        Step(type=StepType.LLM_CALL, model="test", tokens_in=45, tokens_out=25, latency_ms=290),
    ],
    final_output=(
        "I don't know — I can't find any reliable information about a place called "
        "Zarqonia, so I won't guess at a population figure."
    ),
    total_tokens=70,
    total_latency_ms=290,
)
