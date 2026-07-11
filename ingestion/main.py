"""
ingestion/main.py

FastAPI application entrypoint. Assembles db.py (storage) and
routes.py (REST/WebSocket endpoints) into a runnable service, and
creates tables on startup if they don't already exist.

Run directly:
    python -m ingestion.main
or with reload during development:
    uvicorn ingestion.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from . import db as db_module
from ingestion.routes import router

app = FastAPI(
    title="VoltSentinel Ingestion Service",
    description="Receives and persists EV battery telemetry, maintenance tickets, "
                "and BMS command events for the VoltSentinel APM agent.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    conn = db_module.get_connection()
    db_module.init_db(conn)
    conn.close()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "voltsentinel-ingestion"}


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ingestion.main:app", host="0.0.0.0", port=8000, reload=True)