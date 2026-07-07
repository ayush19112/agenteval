"""Static HTML scorecard (Jinja2) — the MVP report surface."""

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from agenteval.evaluators.metrics import scorecard_from_store

DEFAULT_CALIBRATION = Path("judge_calibration.json")


def render_report(store, out_path: str = "report.html", calibration_path: str | None = None) -> str:
    scorecard = scorecard_from_store(store)

    calibration = None
    cal_file = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION
    if cal_file.exists():
        calibration = json.loads(cal_file.read_text(encoding="utf-8"))

    env = Environment(
        loader=PackageLoader("agenteval.report", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("report.html").render(
        groups=scorecard["groups"],
        calibration=calibration,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return str(out)
