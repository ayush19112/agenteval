"""Calibrated LLM-as-judge.

- structured JSON verdicts at temperature 0
- always grades against a reference answer when one exists
- borderline scores trigger extra votes (majority wins)
- "mock" judge models use a deterministic reference-overlap heuristic, so the
  whole pipeline (and the calibration harness) runs offline

Reliability is measured, not assumed: calibration/calibrate.py computes
Cohen's kappa between this judge and human labels on a golden set, and the
scorecard reports it.
"""

import json
import re

from agenteval.core.config import load_config
from agenteval.core.llm import complete
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun
from agenteval.evaluators.base import Check, CheckResult

JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator of AI agent answers. Judge ONLY whether the answer "
    "satisfies the rubric, treating the reference answer (when given) as ground truth. Numeric "
    "values may differ slightly from the reference if the rubric allows approximation. Respond "
    'with ONLY a JSON object: {"passed": true or false, "score": <float 0.0-1.0>, '
    '"reasoning": "<one short sentence>"}'
)

BORDERLINE = (0.35, 0.75)  # scores in this band trigger two extra votes


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")
    return json.loads(match.group())


def _user_prompt(question: str, answer: str, rubric: str, reference: str | None) -> str:
    parts = [
        f"QUESTION:\n{question}",
        f"AGENT ANSWER:\n{answer}",
        f"RUBRIC (what a passing answer must do):\n{rubric}",
    ]
    if reference:
        parts.append(f"REFERENCE ANSWER (ground truth):\n{reference}")
    return "\n\n".join(parts)


async def grade(
    model: str, question: str, answer: str, rubric: str, reference: str | None = None
) -> tuple[bool, float, str]:
    """Grade one answer. Used by the Judge check and by the calibration harness."""
    if model.startswith("mock"):
        return _heuristic_grade(answer, reference)

    async def _vote(temperature: float) -> dict:
        response = await complete(
            model,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _user_prompt(question, answer, rubric, reference)},
            ],
            temperature=temperature,
        )
        return _extract_json(response.content or "")

    first = await _vote(0.0)
    score = float(first.get("score", 0.0))
    if not (BORDERLINE[0] <= score <= BORDERLINE[1]):
        return bool(first.get("passed")), score, str(first.get("reasoning", ""))

    # borderline: majority of three votes, mean score
    votes = [first] + [await _vote(0.3) for _ in range(2)]
    passed = sum(bool(v.get("passed")) for v in votes) >= 2
    mean_score = sum(float(v.get("score", 0.0)) for v in votes) / len(votes)
    return passed, mean_score, f"majority of 3 votes: {votes[0].get('reasoning', '')}"


_WORD = re.compile(r"\d+|[a-z]{3,}")  # numbers of any length + words of 3+ letters


def _heuristic_grade(answer: str, reference: str | None) -> tuple[bool, float, str]:
    """Deterministic offline judge: fraction of reference content words present
    in the answer. Crude on purpose — its whole job is to make the pipeline and
    the calibration math runnable with zero keys."""
    if not reference:
        return False, 0.0, "mock judge requires a reference answer"
    ref_words = set(_WORD.findall(reference.lower()))
    if not ref_words:
        return False, 0.0, "reference has no content words"
    answer_words = set(_WORD.findall(answer.lower()))
    overlap = len(ref_words & answer_words) / len(ref_words)
    return overlap >= 0.6, round(overlap, 3), f"reference-word overlap {overlap:.0%}"


class Judge(Check):
    """Open-ended quality check: an LLM grades the final answer against a rubric."""

    def __init__(self, rubric: str, reference: str | None = None, model: str | None = None):
        self.rubric = rubric
        self.reference = reference  # falls back to task.reference
        self.judge_model = model  # None -> injected by the engine / config

    @property
    def name(self) -> str:
        return f"Judge({self.rubric[:40]})"

    async def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        model = self.judge_model or load_config().judge_model
        reference = self.reference or task.reference
        question = str(task.input.get("prompt", task.input))
        try:
            passed, score, reasoning = await grade(
                model, question, run.final_output, self.rubric, reference
            )
        except Exception as e:
            return CheckResult(
                name=self.name, passed=False, score=0.0, detail=f"judge error: {e}"
            )
        return CheckResult(
            name=self.name, passed=passed, score=max(0.0, min(1.0, score)), detail=reasoning
        )
