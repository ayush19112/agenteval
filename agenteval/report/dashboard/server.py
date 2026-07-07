"""FastAPI app: serves the dashboard payload from SQLite plus the built
React frontend (frontend/dist). Run via `agenteval dashboard`."""

from pathlib import Path

from agenteval.core.store import Store
from agenteval.report.dashboard.data import dashboard_payload

DIST = Path(__file__).parent / "frontend" / "dist"


def create_app(db_path: str, calibration_path: str | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:
        raise RuntimeError(
            "The dashboard needs FastAPI. Install it with: pip install agenteval[dashboard]"
        ) from e

    app = FastAPI(title="agenteval dashboard")

    @app.get("/api/dashboard")
    def dashboard() -> dict:
        store = Store(db_path)
        try:
            return dashboard_payload(store, calibration_path)
        finally:
            store.close()

    if DIST.exists():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
    return app


def serve(db_path: str, calibration_path: str | None = None, port: int = 8501) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            "The dashboard needs uvicorn. Install it with: pip install agenteval[dashboard]"
        ) from e
    if not DIST.exists():
        print(
            "note: frontend not built yet — API only. "
            "Build it with: cd agenteval/report/dashboard/frontend && npm install && npm run build"
        )
    print(f"agenteval dashboard: http://127.0.0.1:{port}  (db: {db_path})")
    uvicorn.run(
        create_app(db_path, calibration_path), host="127.0.0.1", port=port, log_level="warning"
    )
