"""Judge calibration: measure judge/human agreement on the golden set.

Every judge score the platform reports is only as trustworthy as this number.
Run:  python -m agenteval.calibration.calibrate --model gemini/gemini-2.5-flash
      python -m agenteval.calibration.calibrate --model mock/judge   (offline)

Writes judge_calibration.json (read by `agenteval report` to display kappa).
"""

import argparse
import asyncio
import json
from pathlib import Path

from agenteval.evaluators.judge import grade

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
KAPPA_THRESHOLD = 0.6  # below this, judge scores are too noisy to report


def cohens_kappa(human: list[bool], judge: list[bool]) -> float:
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge, strict=True)) / n
    p_human = sum(human) / n
    p_judge = sum(judge) / n
    expected = p_human * p_judge + (1 - p_human) * (1 - p_judge)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def calibrate(model: str, path: Path = GOLDEN_SET) -> dict:
    entries = load_golden_set(path)
    human, judge, disagreements = [], [], []
    for entry in entries:
        passed, score, reasoning = await grade(
            model, entry["question"], entry["answer"], entry["rubric"], entry["reference"]
        )
        human.append(entry["human_passed"])
        judge.append(passed)
        if passed != entry["human_passed"]:
            disagreements.append(
                {"id": entry["id"], "human": entry["human_passed"], "judge": passed,
                 "score": score, "reasoning": reasoning}
            )
    kappa = cohens_kappa(human, judge)
    return {
        "model": model,
        "n": len(entries),
        "agreement": sum(h == j for h, j in zip(human, judge, strict=True)) / len(entries),
        "kappa": round(kappa, 3),
        "threshold": KAPPA_THRESHOLD,
        "calibrated": kappa >= KAPPA_THRESHOLD,
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="judge model (LiteLLM string or mock/judge)")
    parser.add_argument("--golden-set", default=str(GOLDEN_SET))
    parser.add_argument("--out", default="judge_calibration.json")
    args = parser.parse_args()

    report = asyncio.run(calibrate(args.model, Path(args.golden_set)))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "CALIBRATED" if report["calibrated"] else "NOT CALIBRATED — tune the judge prompt"
    print(f"judge={report['model']}  n={report['n']}  agreement={report['agreement']:.1%}  "
          f"kappa={report['kappa']}  [{status}]")
    for d in report["disagreements"]:
        print(f"  disagrees on {d['id']}: human={d['human']} judge={d['judge']} ({d['reasoning']})")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
