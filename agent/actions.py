"""
agent/actions.py

Mocked action functions — the "Act" step of the Perceive -> Reason ->
Decide -> Act loop (§6.1). For the hackathon build these don't call any
real external system; they log a structured record to SQLite (a new
`agent_actions` table, self-contained in this file so Phase 2's
ingestion/db.py doesn't need retroactive changes) and print a
demo-visible line simulating what a live integration would do.

Every function's signature accepts exactly the fields
agent/prompts.py's response schema produces (action_type, priority,
rationale, parameters), so decision_engine.py can dispatch an LLM
decision straight into one of these without any translation layer —
that's the contract this file exists to fulfil.

Table is separate from ingestion's tables on purpose: agent_actions is
this system's own output/audit trail, not ingested telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
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


# ----------------------------------------------------------------------
# The four mocked actions from §6.2's table, plus the explicit no-op
# ----------------------------------------------------------------------
def create_maintenance_ticket(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "medium", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mock predictive-maintenance trigger. A live integration would open a
    real ticket in a fleet-maintenance system; here we log it as 'open'."""
    record = _log_action(
        conn, vehicle_id, "maintenance_trigger", priority, rationale,
        parameters or {}, status="open",
    )
    print(f"[actions] MAINTENANCE TICKET {record['action_id']} opened for {vehicle_id} "
        f"(priority={priority}): {rationale}")
    return record


def recommend_charge_policy(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "medium", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mock charge-discharge recommendation — e.g. cap fast-charge rate or
    limit DoD. A live integration would push this to the vehicle's charge
    controller or the driver app; here we log it as 'recommended'."""
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
    """Mock security escalation — distinct from routine maintenance, flagged
    for fleet-manager review per §7.1. A live integration would page an
    on-call operator; here we log it as 'escalated'."""
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
    """Mock notification. A live integration would send an email/SMS/push;
    here we log it as 'sent'."""
    record = _log_action(
        conn, vehicle_id, "fleet_manager_notification", priority, rationale,
        parameters or {}, status="sent",
    )
    print(f"[actions] NOTIFICATION {record['action_id']} sent for {vehicle_id} "
        f"(priority={priority}): {rationale}")
    return record


def log_no_action(
    conn: sqlite3.Connection, vehicle_id: str, rationale: str,
    priority: str = "low", parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A 'no action warranted' decision is still a decision — logged for
    audit completeness rather than silently discarded, so the fleet view
    can show 'agent reviewed this asset and found nothing concerning'
    instead of no record at all."""
    record = _log_action(
        conn, vehicle_id, "no_action", priority, rationale,
        parameters or {}, status="reviewed",
    )
    return record


# ----------------------------------------------------------------------
# Dispatcher — decision_engine.py calls this once per action the LLM
# returns, so it never needs an if/elif chain over action_type itself.
# ----------------------------------------------------------------------
ACTION_DISPATCH: Dict[str, Callable[..., Dict[str, Any]]] = {
    "maintenance_trigger": create_maintenance_ticket,
    "charge_policy_recommendation": recommend_charge_policy,
    "security_escalation": escalate_incident,
    "fleet_manager_notification": notify_fleet_manager,
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


# ----------------------------------------------------------------------
# Query helpers — consumed by dashboard/ (Phase 5)
# ----------------------------------------------------------------------
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
    # Standalone sanity check: execute one of each action type against a
    # scratch DB, then read them back — including the full-decision path
    # matching prompts.py's few-shot example output exactly.
    import os

    from ingestion.db import get_connection

    test_db = os.path.join("data", "actions_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)

    create_maintenance_ticket(conn, "EVR-0001", "RUL degraded, thermal anomalies recurring",
                            priority="medium", parameters={"reason": "degraded RUL"})
    recommend_charge_policy(conn, "EVR-0002", "Fast-charge frequency well above fleet baseline",
                            priority="low", parameters={"cap_charge_rate": "0.5C"})
    escalate_incident(conn, "EVR-0003", "Unticketed command with GPS mismatch and frequency spike",
                    priority="high", parameters={"unticketed_count": 3})
    notify_fleet_manager(conn, "EVR-0003", "High-priority security escalation raised",
                        priority="high", parameters={})
    log_no_action(conn, "EVR-0004", "All signals nominal, no intervention warranted")

    # Full decision-object path, matching agent/prompts.py's FEW_SHOT_EXAMPLE_OUTPUT shape
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