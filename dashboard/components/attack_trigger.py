from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ingestion.db import insert_command_batch
from ingestion.schemas import CommandEvent
from dashboard.utils import clear_all_caches, get_vehicle_ids
from models.anomaly_detector import DEFAULT_DEPOT_LOCATIONS
from simulator.config import default_config
from agent.auto_alert import notify_fleet_manager_of_attack

SCENARIOS: Dict[str, str] = {
    "no_ticket_gps_mismatch": "Unauthorized command, vehicle in motion (classic Tirri Challenge)",
    "no_ticket_at_depot": "Unauthorized command, GPS at a depot (no-ticket signal alone)",
    "frequency_burst": "Repeated unauthorized commands in a short window (burst)",
}

DEFAULT_SCENARIO = "no_ticket_gps_mismatch"

def _random_road_coords() -> Tuple[float, float]:
    depot = random.choice(DEFAULT_DEPOT_LOCATIONS)
    jitter = default_config.attack_road_gps_jitter_deg
    lat = depot[0] + random.uniform(-jitter, jitter)
    lon = depot[1] + random.uniform(-jitter, jitter)
    return round(lat, 6), round(lon, 6)


def _depot_coords() -> Tuple[float, float]:
    depot = random.choice(DEFAULT_DEPOT_LOCATIONS)
    return round(depot[0], 6), round(depot[1], 6)


def _build_attack_command(vehicle_id: str, timestamp: datetime, at_depot: bool) -> Dict[str, Any]:
    lat, lon = _depot_coords() if at_depot else _random_road_coords()
    return {
        "command_id": f"CMD-{uuid.uuid4().hex[:8].upper()}",
        "vehicle_id": vehicle_id,
        "timestamp": timestamp.isoformat(),
        "command_type": random.choice(list(default_config.attack_command_types)),
        "latitude": lat,
        "longitude": lon,
        "ticket_id": None,
        "is_attack": True,
    }


def build_attack_commands(vehicle_id: str, scenario: str) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cfg = default_config

    if scenario == "no_ticket_at_depot":
        return [_build_attack_command(vehicle_id, now, at_depot=True)]

    if scenario == "frequency_burst":
        burst_count = random.randint(*cfg.attack_burst_command_count)
        commands = []
        for _ in range(burst_count):
            offset_seconds = random.uniform(0, cfg.attack_burst_window_seconds)
            ts = now + timedelta(seconds=offset_seconds)
            commands.append(_build_attack_command(vehicle_id, ts, at_depot=False))
        return commands

    return [_build_attack_command(vehicle_id, now, at_depot=False)]

def inject_attack(conn, vehicle_id: str, scenario: str) -> int:
    raw_commands = build_attack_commands(vehicle_id, scenario)
    validated = [CommandEvent(**c) for c in raw_commands]
    return insert_command_batch(conn, validated)


def inject_attack_and_notify(conn, vehicle_id: str, scenario: str):
    raw_commands = build_attack_commands(vehicle_id, scenario)
    validated = [CommandEvent(**c) for c in raw_commands]
    inserted = insert_command_batch(conn, validated)
    escalation = notify_fleet_manager_of_attack(conn, vehicle_id, raw_commands)
    return inserted, escalation

def render_attack_trigger(conn, default_vehicle_id: Optional[str] = None) -> None:
    st.subheader("Simulate Attack")
    st.caption(
        "Injects a live, Tirri Challenge-style unauthorized BMS command for the "
        "selected vehicle — watch the security anomaly detector and agent decision "
        "layer react on the next refresh."
    )

    vehicle_ids = get_vehicle_ids(conn)
    if not vehicle_ids:
        st.info("No vehicles in the fleet yet — nothing to attack.")
        return

    default_index = (
        vehicle_ids.index(default_vehicle_id) if default_vehicle_id in vehicle_ids else 0
    )

    col1, col2 = st.columns([2, 3])
    vehicle_id = col1.selectbox(
        "Target vehicle", vehicle_ids, index=default_index, key="attack_trigger_vehicle"
    )
    scenario = col2.selectbox(
        "Scenario",
        list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index(DEFAULT_SCENARIO),
        format_func=lambda s: SCENARIOS[s],
        key="attack_trigger_scenario",
    )

    if st.button("🚨 Simulate Attack", key="attack_trigger_button", type="primary"):
        inserted, escalation = inject_attack_and_notify(conn, vehicle_id, scenario)
        st.session_state["pending_vehicle_focus"] = vehicle_id
        st.session_state["map_focus_vehicle"] = vehicle_id

        if escalation:
            st.session_state["dashboard_alert"] = {
                "vehicle_id": vehicle_id,
                "rationale": escalation["rationale"],
                "priority": escalation["priority"],
                "action_id": escalation["action_id"],
            }

        clear_all_caches()
        if escalation:
            st.success(
                f"🚨 Injected {inserted} command(s) for {vehicle_id} — {SCENARIOS[scenario]}. "
                f"Fleet manager notified (escalation {escalation['action_id']})."
            )
        else:
            st.success(f"Injected {inserted} unauthorized command(s) for {vehicle_id}.")
        st.rerun()