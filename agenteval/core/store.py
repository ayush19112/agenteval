"""SQLite persistence — the pivot between `run` and `evaluate`.

`run` writes the runs table; `evaluate` reads runs and (re)writes results;
`report` reads both. Re-scoring never touches the agent.
"""

import json
import sqlite3
import time
from pathlib import Path

from agenteval.core.trace import AgentRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL,
    framework  TEXT NOT NULL,
    model      TEXT NOT NULL,
    category   TEXT NOT NULL,
    sample     INTEGER NOT NULL,
    blob       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id, framework, model);

CREATE TABLE IF NOT EXISTS results (
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    check_name TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    score      REAL NOT NULL,
    detail     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
"""


class Store:
    def __init__(self, path: str | Path = "agenteval.db"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- runs ---------------------------------------------------------------

    def save_run(self, run: AgentRun, category: str, model: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_id,
                run.task_id,
                run.framework,
                model,
                category,
                int(run.metadata.get("sample", 0)),
                run.model_dump_json(),
                time.time(),
            ),
        )
        self._conn.commit()

    def completed_samples(self, task_id: str, framework: str, model: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT sample FROM runs WHERE task_id=? AND framework=? AND model=?",
            (task_id, framework, model),
        )
        return {r[0] for r in rows}

    def load_runs(
        self,
        task_id: str | None = None,
        framework: str | None = None,
        model: str | None = None,
    ) -> list[AgentRun]:
        query, params = "SELECT blob FROM runs WHERE 1=1", []
        for column, value in (("task_id", task_id), ("framework", framework), ("model", model)):
            if value is not None:
                query += f" AND {column}=?"
                params.append(value)
        rows = self._conn.execute(query, params)
        return [AgentRun.model_validate_json(r[0]) for r in rows]

    def run_meta(self) -> dict[str, dict]:
        """run_id -> indexed columns (category/framework/model), for aggregation."""
        rows = self._conn.execute(
            "SELECT run_id, task_id, framework, model, category, created_at FROM runs"
        )
        return {
            r[0]: {
                "task_id": r[1],
                "framework": r[2],
                "model": r[3],
                "category": r[4],
                "created_at": r[5],
            }
            for r in rows
        }

    # -- results ------------------------------------------------------------

    def save_results(self, run_id: str, results: list) -> None:
        """Idempotent: re-evaluating a run replaces its previous results."""
        self._conn.execute("DELETE FROM results WHERE run_id=?", (run_id,))
        self._conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            [(run_id, r.name, int(r.passed), r.score, r.detail) for r in results],
        )
        self._conn.commit()

    def load_results(self) -> dict[str, list[dict]]:
        rows = self._conn.execute("SELECT run_id, check_name, passed, score, detail FROM results")
        out: dict[str, list[dict]] = {}
        for run_id, name, passed, score, detail in rows:
            out.setdefault(run_id, []).append(
                {"name": name, "passed": bool(passed), "score": score, "detail": detail}
            )
        return out

    def export_json(self) -> str:
        """Everything the dashboard needs, as one JSON document."""
        return json.dumps({"meta": self.run_meta(), "results": self.load_results()})
