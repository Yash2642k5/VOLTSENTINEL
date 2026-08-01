from __future__ import annotations

import json
import os
import smtplib
import sqlite3
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional

_CREATE_AGENT_ACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_actions (
    action_id       TEXT PRIMARY KEY,
    vehicle_id      TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    priority        TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    parameters      TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_agent_actions_vehicle ON agent_actions(vehicle_id);"


def init_actions_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_AGENT_ACTIONS_TABLE)
    conn.execute(_CREATE_INDEX)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_action(
    conn: sqlite3.Connection,
    vehicle_id: str,
    action_type: str,
    priority: str,
    rationale: str,
    parameters: Dict[str, Any],
    status: str,
) -> Dict[str, Any]:
    action_id = f"ACT-{uuid.uuid4().hex[:10].upper()}"
    created_at = _now()

    init_actions_table(conn)
    conn.execute(
        """INSERT INTO agent_actions
            (action_id, vehicle_id, action_type, priority, rationale,  parameters, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (action_id, vehicle_id, action_type, priority, rationale,
        json.dumps(parameters), status, created_at),
    )
    conn.commit()

    return {
        "action_id": action_id,
        "vehicle_id": vehicle_id,
        "action_type": action_type,
        "priority": priority,
        "rationale": rationale,
        "parameters": parameters,
        "status": status,
        "created_at": created_at,
    }

def _smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("ALERT_EMAIL_TO"))


def _send_email(subject: str, body: str) -> bool:
    if not _smtp_configured():
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", user or "voltsentinel@localhost")
    to = os.environ["ALERT_EMAIL_TO"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[actions] SMTP send failed ({e}) — falling back to logged-only notification.")
        return False

def create_maintenance_ticket(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "medium", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parameters = dict(parameters or {})
    sent = _send_email(
        subject=f"[VoltSentinel] Maintenance needed — {vehicle_id} ({priority})",
        body=f"Vehicle: {vehicle_id}\nPriority: {priority}\n\n{rationale}",
    )
    parameters["email_sent"] = sent

    record = _log_action(
        conn, vehicle_id, "maintenance_trigger", priority, rationale,
        parameters, status="open",
    )
    print(f"[actions] MAINTENANCE TICKET {record['action_id']} opened for {vehicle_id} "
        f"(priority={priority}, email_sent={sent}): {rationale}")
    return record


def recommend_charge_policy(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "medium", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = _log_action(
        conn, vehicle_id, "charge_policy_recommendation", priority, rationale,
        parameters or {}, status="recommended",
    )
    print(f"[actions] CHARGE POLICY recommendation {record['action_id']} for {vehicle_id} "
        f"(priority={priority}): {rationale}")
    return record


def escalate_incident(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "high", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = _log_action(
        conn, vehicle_id, "security_escalation", priority, rationale,
        parameters or {}, status="escalated",
    )
    print(f"[actions] SECURITY ESCALATION {record['action_id']} for {vehicle_id} "
        f"(priority={priority}): {rationale}")
    return record


def notify_fleet_manager(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "medium", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parameters = dict(parameters or {})
    sent = _send_email(
        subject=f"[VoltSentinel] Alert — {vehicle_id} ({priority})",
        body=f"Vehicle: {vehicle_id}\nPriority: {priority}\n\n{rationale}",
    )
    parameters["email_sent"] = sent

    record = _log_action(
        conn, vehicle_id, "fleet_manager_notification", priority, rationale,
        parameters, status="sent",
    )
    print(f"[actions] NOTIFICATION {record['action_id']} sent for {vehicle_id} "
        f"(priority={priority}, email_sent={sent}): {rationale}")
    return record


def quarantine_vehicle(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "high", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ingestion.db import set_vehicle_quarantine

    parameters = dict(parameters or {})
    record = _log_action(
        conn, vehicle_id, "quarantine_vehicle", priority, rationale,
        parameters, status="active",
    )
    set_vehicle_quarantine(conn, vehicle_id, reason=rationale, action_id=record["action_id"])
    print(f"[actions] QUARANTINE {record['action_id']} imposed on {vehicle_id} "
        f"(priority={priority}): {rationale}")
    return record


def release_vehicle_quarantine(
    conn: sqlite3.Connection, vehicle_id: str, released_by: str,
    rationale: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ingestion.db import release_vehicle_quarantine as _release

    result = _release(conn, vehicle_id, released_by=released_by)
    if result is None:
        raise ValueError(f"{vehicle_id} is not currently quarantined")

    rationale = rationale or f"Quarantine manually released by {released_by}."
    record = _log_action(
        conn, vehicle_id, "quarantine_release", "low", rationale,
        parameters or {"released_by": released_by}, status="released",
    )
    print(f"[actions] QUARANTINE RELEASED for {vehicle_id} by {released_by}")
    return record


def log_no_action(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "low", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = _log_action(
        conn, vehicle_id, "no_action", priority, rationale,
        parameters or {}, status="reviewed",
    )
    return record

ACTION_DISPATCH: Dict[str, Callable[..., Dict[str, Any]]] = {
    "maintenance_trigger": create_maintenance_ticket,
    "charge_policy_recommendation": recommend_charge_policy,
    "security_escalation": escalate_incident,
    "fleet_manager_notification": notify_fleet_manager,
    "quarantine_vehicle": quarantine_vehicle,
    "no_action": log_no_action,
}


def execute_action(
    conn: sqlite3.Connection,
    vehicle_id: str,
    action_type: str,
    priority: str,
    rationale: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single entrypoint matching agent/prompts.py's response schema field
    names exactly (action_type, priority, rationale, parameters)."""
    handler = ACTION_DISPATCH.get(action_type)
    if handler is None:
        raise ValueError(
            f"Unknown action_type '{action_type}' — must be one of {list(ACTION_DISPATCH)}"
        )
    return handler(conn, vehicle_id, rationale, priority, parameters)


def execute_decision(conn: sqlite3.Connection, decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Takes one parsed LLM decision matching prompts.py's full response
    schema ({"vehicle_id": ..., "actions": [...], "summary": ...}) and
    executes every action in it, returning the resulting audit records."""
    vehicle_id = decision["vehicle_id"]
    results = []
    for action in decision.get("actions", []):
        results.append(execute_action(
            conn, vehicle_id,
            action["action_type"], action["priority"],
            action["rationale"], action.get("parameters"),
        ))
    return results

def get_actions_for_vehicle(conn: sqlite3.Connection, vehicle_id: str) -> List[sqlite3.Row]:
    init_actions_table(conn)
    return conn.execute(
        "SELECT * FROM agent_actions WHERE vehicle_id = ? ORDER BY created_at DESC", (vehicle_id,)
    ).fetchall()


def get_all_actions(conn: sqlite3.Connection, exclude_no_action: bool = False) -> List[sqlite3.Row]:
    init_actions_table(conn)
    query = "SELECT * FROM agent_actions"
    if exclude_no_action:
        query += " WHERE action_type != 'no_action'"
    query += " ORDER BY created_at DESC"
    return conn.execute(query).fetchall()


if __name__ == "__main__":
    import os as _os

    from ingestion.db import get_connection, init_db, is_vehicle_quarantined

    test_db = os.path.join("data", "actions_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    create_maintenance_ticket(conn, "EVR-0001", "RUL degraded, thermal anomalies recurring",
                            priority="medium", parameters={"reason": "degraded RUL"})
    recommend_charge_policy(conn, "EVR-0002", "Fast-charge frequency well above fleet baseline",
                            priority="low", parameters={"cap_charge_rate": "0.5C"})
    escalate_incident(conn, "EVR-0003", "Unticketed command with GPS mismatch and frequency spike",
                    priority="high", parameters={"unticketed_count": 3})
    notify_fleet_manager(conn, "EVR-0003", "High-priority security escalation raised",
                        priority="high", parameters={})
    log_no_action(conn, "EVR-0004", "All signals nominal, no intervention warranted")

    # Tier 3 — impose then release a quarantine
    quarantine_vehicle(conn, "EVR-0003", "Repeated unticketed commands with GPS mismatch",
                        priority="high", parameters={"unticketed_count": 3})
    print(f"\nEVR-0003 quarantined: {is_vehicle_quarantined(conn, 'EVR-0003')}")
    release_vehicle_quarantine(conn, "EVR-0003", released_by="ops_manager_demo")
    print(f"EVR-0003 quarantined after release: {is_vehicle_quarantined(conn, 'EVR-0003')}")

    decision = {
        "vehicle_id": "EVR-0042",
        "actions": [
            {
                "action_type": "maintenance_trigger",
                "priority": "medium",
                "rationale": "RUL degraded at 76.2% capacity with 42 cycles projected remaining, "
                            "plus 3 recurring thermal anomalies.",
                "parameters": {"reason": "degraded RUL with recurring thermal anomalies"},
            },
            {
                "action_type": "no_action",
                "priority": "low",
                "rationale": "Charging behaviour only mildly elevated vs fleet baseline and stable.",
                "parameters": {},
            },
        ],
        "summary": "EVR-0042 needs scheduled maintenance soon; charging is fine for now.",
    }
    results = execute_decision(conn, decision)
    print(f"\nexecute_decision() returned {len(results)} action records for EVR-0042")

    print(f"\nAll logged actions (excluding no_action):")
    for row in get_all_actions(conn, exclude_no_action=True):
        print(f"  {row['action_id']} | {row['vehicle_id']} | {row['action_type']} "
            f"| {row['priority']} | {row['status']}")

    print(f"\nActions for EVR-0003 specifically:")
    for row in get_actions_for_vehicle(conn, "EVR-0003"):
        print(f"  {row['action_id']} | {row['action_type']} | {row['rationale']}")

    conn.close()
    os.remove(test_db)