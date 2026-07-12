"""
dashboard/components/attack_trigger.py

The live "Simulate Attack" demo trigger (project doc §5.1's Presentation
row, §8's "allows the live demo to trigger a realistic scenario on
demand"). Distinct from simulator/attack_injector.py, which generates a
whole fleet's worth of attack commands anchored to a *simulated historical*
time window for batch runs — this component injects ONE scenario's worth
of unauthorized BMS command(s) anchored to *now*, directly into the live
SQLite DB, so the fleet map, alert feed, and agent recommendations panel
all react to it on the very next rerun.

Reuses attack_injector.py's mechanics and simulator/config.py's tuned
parameters (GPS jitter degrees, burst count/window) rather than
hand-picking new magic numbers here — the injected traffic should look
like the same Tirri Challenge pattern the rest of the system was tuned
against, just fired on demand instead of at a random point in a 500-cycle
history.

Writes straight through ingestion/db.py's insert_command_batch (same
idempotent insert path routes.py's REST/WS endpoints use), validated
through ingestion/schemas.py first — so an injected demo command can
never reach SQLite in a shape the rest of the pipeline wouldn't accept
from a real client.

Two-step split, matching agent_recommendations.py's "preview vs commit"
pattern:
    build_attack_commands(...) -> pure, no st.*, no DB writes. Testable.
    inject_attack(...)         -> validates + writes + returns count.
    render_attack_trigger(...) -> the only function here that touches st.*.
"""

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

# ----------------------------------------------------------------------
# Scenario definitions — each maps directly to one of the §7.1 signals
# (no ticket, GPS mismatch, frequency spike) so the detector's reaction
# to each button press is explainable in the demo narrative.
# ----------------------------------------------------------------------
SCENARIOS: Dict[str, str] = {
    "no_ticket_gps_mismatch": "Unauthorized command, vehicle in motion (classic Tirri Challenge)",
    "no_ticket_at_depot": "Unauthorized command, GPS at a depot (no-ticket signal alone)",
    "frequency_burst": "Repeated unauthorized commands in a short window (burst)",
}

DEFAULT_SCENARIO = "no_ticket_gps_mismatch"


# ----------------------------------------------------------------------
# Pure builders — no st.*, no DB access. Mirrors attack_injector.py's
# _single_attack_command / _generate_vehicle_attacks mechanics.
# ----------------------------------------------------------------------
def _random_road_coords() -> Tuple[float, float]:
    """A depot as a rough regional anchor, jittered far enough away to
    represent 'in motion on a public road' — same idea as
    attack_injector.AttackInjector._random_road_coords()."""
    depot = random.choice(DEFAULT_DEPOT_LOCATIONS)
    jitter = default_config.attack_road_gps_jitter_deg
    lat = depot[0] + random.uniform(-jitter, jitter)
    lon = depot[1] + random.uniform(-jitter, jitter)
    return round(lat, 6), round(lon, 6)


def _depot_coords() -> Tuple[float, float]:
    """GPS exactly at a depot but still with no ticket — isolates the
    no_ticket_flag signal on its own, without gps_mismatch_flag also
    firing, per attack_injector.py's 'keeps the detector honest' comment."""
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
        # Ground-truth marker only — same convention as attack_injector.py.
        # The anomaly detector must never read this field; it exists purely
        # so a later "was this flagged correctly?" check could use it.
        "is_attack": True,
    }


def build_attack_commands(vehicle_id: str, scenario: str) -> List[Dict[str, Any]]:
    """Builds the raw command dict(s) for one scenario, anchored to now.
    Does not touch the DB — kept separate so this is unit-testable
    without a live sqlite connection or a running Streamlit app."""
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

    # Default / "no_ticket_gps_mismatch": the classic single Tirri Challenge event.
    return [_build_attack_command(vehicle_id, now, at_depot=False)]


# ----------------------------------------------------------------------
# DB write path — validates through schemas.py, inserts via db.py
# ----------------------------------------------------------------------
def inject_attack(conn, vehicle_id: str, scenario: str) -> int:
    """Builds, validates, and writes the scenario's command(s). Returns
    the number of rows actually inserted (insert_command_batch already
    de-dupes on the command_id primary key, so this also matches the
    idempotency guarantee the rest of the ingestion layer relies on)."""
    raw_commands = build_attack_commands(vehicle_id, scenario)
    validated = [CommandEvent(**c) for c in raw_commands]
    return insert_command_batch(conn, validated)


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_attack_trigger(conn, default_vehicle_id: Optional[str] = None) -> None:
    """Renders the vehicle/scenario picker and the trigger button.
    default_vehicle_id lets app.py pre-select whichever asset is
    currently open in the detail view (from fleet_map.py's selection),
    so the demo can attack "the vehicle we're already looking at" in
    one click instead of two."""
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
        inserted = inject_attack(conn, vehicle_id, scenario)
        clear_all_caches()
        st.success(
            f"Injected {inserted} unauthorized command(s) for {vehicle_id} "
            f"— {SCENARIOS[scenario]}. Refreshing fleet view..."
        )
        st.rerun()