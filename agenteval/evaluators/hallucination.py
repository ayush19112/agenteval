"""Hallucination measurement, two complementary signals:

- Faithfulness: decompose the final answer into atomic claims and check each
  against the evidence available in the trace (tool outputs + the task
  reference). Score = supported / total. LLM-judged for real models, with a
  deterministic lexical fallback for mock/offline runs — both flow through the
  same calibration discipline as the judge.
- Abstained: on unanswerable tasks the correct behavior is refusal. A
  confident fabricated answer is a hallucination; this check is programmatic.
"""

import re

from agenteval.core.config import load_config
from agenteval.core.llm import complete
from agenteval.core.task import Task
from agenteval.core.trace import AgentRun, StepType
from agenteval.evaluators.base import Check, CheckResult
from agenteval.evaluators.judge import _WORD, _extract_json

FAITHFULNESS_SYSTEM = (
    "You are a fact-checking evaluator. Decompose the ANSWER into its atomic factual claims, "
    "then mark each claim supported=true only if it is entailed by the EVIDENCE. Respond with "
    'ONLY JSON: {"claims": [{"text": "<claim>", "supported": true or false}, ...]}. '
    "Statements that make no factual assertion (greetings, hedges) are not claims."
)


def trace_evidence(run: AgentRun, task: Task) -> list[str]:
    evidence = [
        s.tool_output for s in run.steps if s.type == StepType.TOOL_RESULT and s.tool_output
    ]
    if task.reference:
        evidence.append(task.reference)
    return evidence


class Faithfulness(Check):
    """Fraction of the answer's claims supported by trace evidence."""

    def __init__(self, threshold: float = 0.7, model: str | None = None):
        self.threshold = threshold
        self.judge_model = model  # None -> injected by the engine / config

    @property
    def name(self) -> str:
        return "Faithfulness"

    async def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        if not run.final_output.strip():
            return CheckResult(name=self.name, passed=False, score=0.0, detail="empty output")
        model = self.judge_model or load_config().judge_model
        evidence = trace_evidence(run, task)
        try:
            if model.startswith("mock"):
                claims = _lexical_claims(run.final_output, evidence)
            else:
                claims = await _llm_claims(model, run.final_output, evidence)
        except Exception as e:
            return CheckResult(
                name=self.name, passed=False, score=0.0, detail=f"faithfulness error: {e}"
            )
        if not claims:
            return CheckResult(name=self.name, passed=True, score=1.0, detail="no factual claims")
        supported = sum(1 for c in claims if c["supported"])
        score = supported / len(claims)
        unsupported = [c["text"][:80] for c in claims if not c["supported"]]
        detail = f"{supported}/{len(claims)} claims supported"
        if unsupported:
            detail += f"; unsupported: {unsupported}"
        return CheckResult(
            name=self.name, passed=score >= self.threshold, score=round(score, 3), detail=detail
        )


async def _llm_claims(model: str, answer: str, evidence: list[str]) -> list[dict]:
    prompt = "ANSWER:\n" + answer + "\n\nEVIDENCE:\n" + "\n---\n".join(evidence or ["(none)"])
    response = await complete(
        model,
        [
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    data = _extract_json(response.content or "")
    return [
        {"text": str(c.get("text", "")), "supported": bool(c.get("supported"))}
        for c in data.get("claims", [])
    ]


_CLAIM_SPLIT = re.compile(r"[.;\n](?!\d)|,\s+(?:and|but|while)\s+", re.IGNORECASE)


def _lexical_claims(answer: str, evidence: list[str]) -> list[dict]:
    """Offline fallback: split into clauses; a claim is supported when most of
    its content words appear in the evidence AND every number it states does
    (numbers are the load-bearing facts in eval answers)."""
    evidence_words = set(_WORD.findall(" ".join(evidence).lower()))
    claims = []
    for clause in _CLAIM_SPLIT.split(answer):
        words = _WORD.findall(clause.lower())
        if len(words) < 2:
            continue
        overlap = sum(1 for w in words if w in evidence_words) / len(words)
        numbers_ok = all(w in evidence_words for w in words if w.isdigit())
        claims.append({"text": clause.strip(), "supported": overlap >= 0.6 and numbers_ok})
    return claims


_ABSTAIN_MARKERS = (
    "don't know",
    "do not know",
    "don't have",
    "do not have",
    "can't answer",
    "cannot answer",
    "can't give",
    "cannot give",
    "cannot determine",
    "can't determine",
    "no reliable information",
    "not able to",
    "unable to",
    "won't guess",
    "not sure",
    "couldn't find",
    "could not find",
    "couldn't retrieve",
    "could not retrieve",
)


class Abstained(Check):
    """Passes when the agent explicitly declines instead of fabricating."""

    def evaluate(self, run: AgentRun, task: Task) -> CheckResult:
        text = run.final_output.lower()
        matched = [m for m in _ABSTAIN_MARKERS if m in text]
        return CheckResult(
            name=self.name,
            passed=bool(matched),
            score=1.0 if matched else 0.0,
            detail=f"abstention markers: {matched}" if matched else "answered confidently",
        )
