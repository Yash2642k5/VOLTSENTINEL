from __future__ import annotations

import sqlite3
from typing import Generator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from . import db as db_module
from .schemas import (
    CommandBatch,
    CommandEvent,
    IngestionAck,
    MaintenanceBatch,
    MaintenanceTicket,
    StreamMessage,
    StreamMessageType,
    TelemetryBatch,
    TelemetryReading,
)

router = APIRouter()

def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = db_module.get_connection()
    try:
        yield conn
    finally:
        conn.close()

@router.post("/telemetry", response_model=IngestionAck, tags=["ingestion"])
def ingest_telemetry(batch: TelemetryBatch, conn: sqlite3.Connection = Depends(get_db)) -> IngestionAck:
    try:
        inserted = db_module.insert_telemetry_batch(conn, batch.readings)
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"database error: {e}")
    return IngestionAck(records_received=len(batch.readings), records_inserted=inserted)


@router.post("/maintenance", response_model=IngestionAck, tags=["ingestion"])
def ingest_maintenance(batch: MaintenanceBatch, conn: sqlite3.Connection = Depends(get_db)) -> IngestionAck:
    try:
        inserted = db_module.insert_maintenance_batch(conn, batch.tickets)
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"database error: {e}")
    return IngestionAck(records_received=len(batch.tickets), records_inserted=inserted)


@router.post("/commands", response_model=IngestionAck, tags=["ingestion"])
def ingest_commands(batch: CommandBatch, conn: sqlite3.Connection = Depends(get_db)) -> IngestionAck:
    try:
        inserted = db_module.insert_command_batch(conn, batch.commands)
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=400, detail=f"integrity error (check ticket_id exists): {e}")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"database error: {e}")
    return IngestionAck(records_received=len(batch.commands), records_inserted=inserted)

@router.get("/vehicles", response_model=List[str], tags=["query"])
def list_vehicles(conn: sqlite3.Connection = Depends(get_db)) -> List[str]:
    return db_module.get_all_vehicle_ids(conn)


@router.get("/vehicles/{vehicle_id}/telemetry", tags=["query"])
def vehicle_telemetry(vehicle_id: str, conn: sqlite3.Connection = Depends(get_db)) -> List[dict]:
    rows = db_module.get_telemetry_for_vehicle(conn, vehicle_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no telemetry found for {vehicle_id}")
    return [dict(r) for r in rows]


@router.get("/vehicles/{vehicle_id}/tickets", tags=["query"])
def vehicle_tickets(vehicle_id: str, conn: sqlite3.Connection = Depends(get_db)) -> List[dict]:
    return [dict(r) for r in db_module.get_tickets_for_vehicle(conn, vehicle_id)]


@router.get("/vehicles/{vehicle_id}/commands", tags=["query"])
def vehicle_commands(vehicle_id: str, conn: sqlite3.Connection = Depends(get_db)) -> List[dict]:
    return [dict(r) for r in db_module.get_commands_for_vehicle(conn, vehicle_id)]


@router.get("/commands/unticketed", tags=["query"])
def unticketed_commands(
    vehicle_id: Optional[str] = None, conn: sqlite3.Connection = Depends(get_db)
) -> List[dict]:
    """Commands with no matching maintenance ticket — the security-anomaly
    precondition from the project doc (section 7.1)."""
    return [dict(r) for r in db_module.get_unticketed_commands(conn, vehicle_id)]


@router.get("/stats", tags=["query"])
def stats(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    return db_module.row_counts(conn)

# WebSocket — live single-event streaming
_STREAM_HANDLERS = {
    StreamMessageType.TELEMETRY: (TelemetryReading, db_module.insert_telemetry),
    StreamMessageType.MAINTENANCE: (MaintenanceTicket, db_module.insert_maintenance_ticket),
    StreamMessageType.COMMAND: (CommandEvent, db_module.insert_command),
}


@router.websocket("/ws/stream")
async def stream_events(websocket: WebSocket) -> None:
    await websocket.accept()
    conn = db_module.get_connection()
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = StreamMessage(**raw)
                model_cls, insert_fn = _STREAM_HANDLERS[message.type]
                record = model_cls(**message.payload)

                before = conn.total_changes
                insert_fn(conn, record)
                conn.commit()
                inserted = conn.total_changes - before

                await websocket.send_json(
                    IngestionAck(records_received=1, records_inserted=inserted).model_dump()
                )
            except WebSocketDisconnect:
                raise
            except Exception as e:
                await websocket.send_json(
                    IngestionAck(status="error", records_received=1, errors=[str(e)]).model_dump()
                )
    except WebSocketDisconnect:
        pass
    finally:
        conn.close()