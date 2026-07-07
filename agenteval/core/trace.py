"""The normalized trace format — the contract every adapter emits and every
evaluator reads. Nothing downstream ever touches framework internals."""

from enum import StrEnum

from pydantic import BaseModel, Field


class StepType(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class Step(BaseModel):
    type: StepType
    content: str | None = None
    # llm_call fields
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    # tool fields
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_output: str | None = None
    latency_ms: float = 0.0
    timestamp: float = 0.0


class AgentRun(BaseModel):
    task_id: str
    run_id: str  # unique per sample: f"{task_id}:{sample}:{uuid4}"
    framework: str  # "raw" | "langgraph" | "crewai"
    input: dict
    steps: list[Step] = Field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None
    metadata: dict = Field(default_factory=dict)
