"""Deterministic checks — cheap, reliable, and preferred over the judge
wherever they apply. All read only the normalized AgentRun."""

import json
import re

from agenteval.core.task import Task
from agenteval.core.trace import AgentRun, StepType
from agenteval.evaluators.base import Check, CheckResult


def _result(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, score=1.0 if passed else 0.0, detail=detail)


class ExactMatch(Check):
    def __init__(self, expected: str):
        self.expected = expected

    @property
    def name(self) -> str:
        return f"ExactMatch({self.expected!r})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        passed = run.final_output.strip() == self.expected.strip()
        return _result(self.name, passed, f"got: {run.final_output[:120]!r}")


class Contains(Check):
    def __init__(self, text: str, case_sensitive: bool = False):
        self.text = text
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return f"Contains({self.text!r})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        haystack = run.final_output if self.case_sensitive else run.final_output.lower()
        needle = self.text if self.case_sensitive else self.text.lower()
        return _result(self.name, needle in haystack, f"got: {run.final_output[:120]!r}")


class NotContains(Check):
    """Output must NOT contain the text — e.g. a prompt-injection payload."""

    def __init__(self, text: str, case_sensitive: bool = False):
        self.text = text
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return f"NotContains({self.text!r})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        haystack = run.final_output if self.case_sensitive else run.final_output.lower()
        needle = self.text if self.case_sensitive else self.text.lower()
        return _result(self.name, needle not in haystack, f"got: {run.final_output[:120]!r}")


class Regex(Check):
    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self.pattern = pattern
        self._compiled = re.compile(pattern, flags)

    @property
    def name(self) -> str:
        return f"Regex({self.pattern!r})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        passed = self._compiled.search(run.final_output) is not None
        return _result(self.name, passed, f"got: {run.final_output[:120]!r}")


class JSONSchema(Check):
    """Final output must parse as JSON and match a minimal schema
    (type / required / properties — enough for eval outputs, no extra deps)."""

    def __init__(self, schema: dict):
        self.schema = schema

    @property
    def name(self) -> str:
        return "JSONSchema"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        try:
            data = json.loads(run.final_output)
        except json.JSONDecodeError as e:
            return _result(self.name, False, f"not valid JSON: {e}")
        error = _validate(data, self.schema, path="$")
        return _result(self.name, error is None, error or "valid")


_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _validate(data, schema: dict, path: str) -> str | None:
    expected = schema.get("type")
    if expected and not isinstance(data, _JSON_TYPES[expected]):
        return f"{path}: expected {expected}, got {type(data).__name__}"
    if expected == "object":
        for key in schema.get("required", []):
            if key not in data:
                return f"{path}: missing required key '{key}'"
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                error = _validate(data[key], sub, f"{path}.{key}")
                if error:
                    return error
    if expected == "array" and "items" in schema:
        for i, item in enumerate(data):
            error = _validate(item, schema["items"], f"{path}[{i}]")
            if error:
                return error
    return None


class ToolCalled(Check):
    """The agent must have called `tool_name`; optionally with a matching
    subset of args, and/or at a given 0-based position among tool calls."""

    def __init__(self, tool_name: str, args: dict | None = None, order: int | None = None):
        self.tool_name = tool_name
        self.args = args
        self.order = order

    @property
    def name(self) -> str:
        return f"ToolCalled({self.tool_name})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        calls = [s for s in run.steps if s.type == StepType.TOOL_CALL]
        matches = [
            (i, s)
            for i, s in enumerate(calls)
            if s.tool_name == self.tool_name
            and (self.args is None or self.args.items() <= (s.tool_args or {}).items())
        ]
        if not matches:
            called = [s.tool_name for s in calls]
            return _result(self.name, False, f"not called (tools called: {called})")
        if self.order is not None and all(i != self.order for i, _ in matches):
            return _result(
                self.name, False, f"called at position(s) {[i for i, _ in matches]}, "
                f"expected {self.order}"
            )
        return _result(self.name, True, f"called with {matches[0][1].tool_args}")


class ToolNotCalled(Check):
    """The agent must NOT have called `tool_name` (adversarial tasks)."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    @property
    def name(self) -> str:
        return f"ToolNotCalled({self.tool_name})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        called = any(
            s.type == StepType.TOOL_CALL and s.tool_name == self.tool_name for s in run.steps
        )
        return _result(self.name, not called)


class StepCountMax(Check):
    """Efficiency bound: at most `n` agent actions (LLM calls + tool calls)."""

    def __init__(self, n: int):
        self.n = n

    @property
    def name(self) -> str:
        return f"StepCountMax({self.n})"

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        actions = sum(
            1 for s in run.steps if s.type in (StepType.LLM_CALL, StepType.TOOL_CALL)
        )
        return _result(self.name, actions <= self.n, f"{actions} actions")


class OutputNonEmpty(Check):
    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        return _result(self.name, bool(run.final_output.strip()))
