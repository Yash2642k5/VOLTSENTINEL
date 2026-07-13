"""
agent/auto_alert.py

Fires automatically the moment attack_trigger.py injects a live event —
no LLM call, no human clicking "Run agent reasoning". Runs the real
rule-based AnomalyDetector against the vehicle's command history,
scoped to just the commands that were just injected, and logs a
security_escalation + fleet_manager_notification via actions.py so it
shows up in the alert feed on the very next rerun.

Deliberately separate from agent/decision_engine.py's LLM reasoning:
this is the always-on floor, not a replacement for it. If/when
decision_engine.py's richer reasoning is available, it can run on top
of this — but the fleet manager should never be silently unnotified
just because Gemini is unavailable or slow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def notify_fleet_manager_of_attack(
    conn, vehicle_id: str, injected_commands: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    from ingestion.db import get_commands_for_vehicle
    from models.anomaly_detector import AnomalyDetector
    from agent.actions import escalate_incident, notify_fleet_manager

    injected_ids = {c["command_id"] for c in injected_commands}
    if not injected_ids:
        return None

    rows = get_commands_for_vehicle(conn, vehicle_id)
    cmd_df = pd.DataFrame([dict(r) for r in rows])
    result = AnomalyDetector().detect_command_anomalies(
        cmd_df.drop(columns=["is_attack"], errors="ignore")
    )

    new_flagged = result[result["command_id"].isin(injected_ids) & (result["signal_count"] >= 1)]
    if new_flagged.empty:
        return None

    worst = new_flagged.loc[new_flagged["signal_count"].idxmax()]
    signals = []
    if worst["no_ticket_flag"]:
        signals.append("no matching maintenance ticket")
    if worst["gps_mismatch_flag"]:
        signals.append("GPS inconsistent with any known depot")
    if worst["frequency_spike_flag"]:
        signals.append("command frequency spike")

    rationale = (
        f"{len(new_flagged)} unauthorized BMS command(s) detected for {vehicle_id}: "
        f"{', '.join(signals)}. Severity: {worst['severity']}."
    )
    priority = "high" if worst["severity"] == "high" else "medium"

    escalation = escalate_incident(
        conn, vehicle_id, rationale, priority=priority,
        parameters={"signal_count": int(worst["signal_count"]), "severity": worst["severity"]},
    )
    notify_fleet_manager(
        conn, vehicle_id,
        f"Security escalation raised for {vehicle_id} — see incident {escalation['action_id']} for details.",
        priority=priority,
        parameters={"related_action_id": escalation["action_id"]},
    )
    return escalation