"""
dashboard/utils.py

Shared helpers for every dashboard/components/*.py file and app.py:
cached DB access, cached fleet-profile computation, and the formatting/
color-mapping functions used to render badges consistently across the
fleet view, health charts, alert feed, and agent recommendations panel.

Nothing in here renders a widget itself (no st.write/st.metric/etc.) —
this is pure data-and-formatting, matching the project's existing split
between "compute" (models/, agent/) and "display" (dashboard/). Keeping
that split here too means components/*.py stays thin and every number
on screen traces back to one of the query functions below.

Caching notes:
  - get_connection() is a single cached resource (st.cache_resource) —
    sqlite3.Connection isn't picklable/hashable in the way st.cache_data
    wants, and there's no reason to reopen the DB file every rerun.
  - Every function that takes the connection as an argument names the
    parameter `_conn` (leading underscore) — this is Streamlit's
    convention for "don't try to hash this argument," required for any
    unhashable object passed into a cache_data-wrapped function.
  - TTLs are short (a few seconds) rather than infinite: this dashboard
    is built around a live "Simulate Attack" demo trigger (Phase 5,
    dashboard/components/attack_trigger.py), so a fleet view that never
    refreshes would defeat the point of the demo.

Plain-language labels: STATUS_PLAIN_LABELS / SECURITY_PLAIN_LABELS (and
their status_plain_label() / security_plain_label() lookup helpers) exist
specifically for a non-technical fleet-manager audience, alongside the
existing technical labels (status_label(), etc.) which some components
still use for badges/logs.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

DEFAULT_CACHE_TTL_SECONDS = 8

# ----------------------------------------------------------------------
# Color / label mappings — the single source of truth every component
# should import from, instead of hardcoding hex codes or label strings.
# ----------------------------------------------------------------------
RISK_LEVEL_COLORS: Dict[str, str] = {
    "minimal": "#2E7D32",  # green
    "low": "#9E9D24",      # olive
    "medium": "#F57C00",   # amber
    "high": "#C62828",     # red
}
RISK_LEVEL_ORDER = ("minimal", "low", "medium", "high")

RUL_STATUS_COLORS: Dict[str, str] = {
    "healthy": "#2E7D32",
    "watch": "#9E9D24",
    "degraded": "#F57C00",
    "critical": "#C62828",
    "insufficient_data": "#757575",
    "no_fit": "#757575",
}

SECURITY_SEVERITY_COLORS: Dict[str, str] = {
    "none": "#2E7D32",
    "low": "#9E9D24",
    "medium": "#F57C00",
    "high": "#C62828",
}

PRIORITY_COLORS: Dict[str, str] = {
    "low": "#757575",
    "medium": "#F57C00",
    "high": "#D84315",
    "critical": "#C62828",
}

ACTION_TYPE_LABELS: Dict[str, str] = {
    "maintenance_trigger": "Maintenance Trigger",
    "charge_policy_recommendation": "Charge Policy Recommendation",
    "security_escalation": "Security Escalation",
    "fleet_manager_notification": "Fleet Manager Notification",
    "no_action": "No Action",
}

# ----------------------------------------------------------------------
# Plain-language labels — for a non-technical fleet-manager audience.
# ----------------------------------------------------------------------
STATUS_PLAIN_LABELS: Dict[str, str] = {
    "healthy": "Battery is healthy",
    "watch": "Keep an eye on this one",
    "degraded": "Needs a checkup soon",
    "critical": "Needs attention now",
    "insufficient_data": "Not enough data yet",
    "no_fit": "Not enough data yet",
}

SECURITY_PLAIN_LABELS: Dict[str, str] = {
    "none": "No suspicious activity",
    "low": "Minor irregularity noticed",
    "medium": "Suspicious command detected",
    "high": "Likely unauthorized tampering",
}


def status_plain_label(value: Optional[str]) -> str:
    if not value:
        return "Status unknown"
    return STATUS_PLAIN_LABELS.get(value, "Status unknown")


def security_plain_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    return SECURITY_PLAIN_LABELS.get(value, "Unknown")


# ----------------------------------------------------------------------
# Connection — one cached resource, reused across reruns
# ----------------------------------------------------------------------
@st.cache_resource
def get_connection() -> sqlite3.Connection:
    from ingestion.db import get_connection as _get_connection, init_db

    conn = _get_connection()
    init_db(conn)
    return conn


def ensure_seeded(_conn: sqlite3.Connection) -> None:
    """Checked on every rerun (cheap — one COUNT query), not just once per
    process lifetime. Streamlit Community Cloud's ephemeral disk can lose
    data/voltsentinel.db without necessarily killing the Python process/
    cache_resource cache, so gating this behind get_connection() alone
    isn't reliable — this catches an underlying-file wipe mid-session too.
    Reseeds automatically if the DB comes back empty; no-ops otherwise."""
    from ingestion.db import row_counts, DB_PATH

    if row_counts(_conn)["telemetry"] == 0:
        from scripts.seed_db import seed as _seed_db
        _seed_db(fleet_size=50, num_cycles=500, random_seed=42, db_path=DB_PATH, wipe=False)
        clear_all_caches()  # force every @st.cache_data query to reread the freshly seeded rows


def clear_all_caches() -> None:
    """Called by attack_trigger.py (Phase 5) after injecting a live event,
    so the fleet view reflects it on the very next rerun instead of
    waiting out the TTL."""
    st.cache_data.clear()


# ----------------------------------------------------------------------
# Cached queries — fleet-level
# ----------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_ids(_conn: sqlite3.Connection) -> List[str]:
    from ingestion.db import get_all_vehicle_ids

    return get_all_vehicle_ids(_conn)


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_fleet_profile(_conn: sqlite3.Connection) -> pd.DataFrame:
    """The dashboard's single most expensive call — fits RUL for every
    vehicle, runs both anomaly detectors, and scores charging behaviour.
    Cached so switching between dashboard tabs doesn't redo this work."""
    from models.risk_engine import RiskEngine

    engine = RiskEngine()
    return engine.build_fleet_profile(_conn)


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_recent_actions(_conn: sqlite3.Connection, exclude_no_action: bool = True) -> pd.DataFrame:
    """Backs the alert feed (dashboard/components/alert_feed.py, Phase 5).
    Returns agent_actions rows as a DataFrame, newest first."""
    from agent.actions import get_all_actions

    rows = get_all_actions(_conn, exclude_no_action=exclude_no_action)
    if not rows:
        return pd.DataFrame(columns=[
            "action_id", "vehicle_id", "action_type", "priority",
            "rationale", "parameters", "status", "created_at",
        ])
    return pd.DataFrame([dict(r) for r in rows])


# ----------------------------------------------------------------------
# Cached queries — per-vehicle (health_chart.py, fleet_map.py, etc.)
# ----------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_telemetry(_conn: sqlite3.Connection, vehicle_id: str) -> pd.DataFrame:
    from ingestion.db import get_telemetry_for_vehicle

    rows = get_telemetry_for_vehicle(_conn, vehicle_id)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_commands(_conn: sqlite3.Connection, vehicle_id: str) -> pd.DataFrame:
    from ingestion.db import get_commands_for_vehicle

    rows = get_commands_for_vehicle(_conn, vehicle_id)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_tickets(_conn: sqlite3.Connection, vehicle_id: str) -> pd.DataFrame:
    from ingestion.db import get_tickets_for_vehicle

    rows = get_tickets_for_vehicle(_conn, vehicle_id)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_actions(_conn: sqlite3.Connection, vehicle_id: str) -> pd.DataFrame:
    from agent.actions import get_actions_for_vehicle

    rows = get_actions_for_vehicle(_conn, vehicle_id)
    if not rows:
        return pd.DataFrame(columns=[
            "action_id", "vehicle_id", "action_type", "priority",
            "rationale", "parameters", "status", "created_at",
        ])
    return pd.DataFrame([dict(r) for r in rows])


def get_vehicle_row(profile_df: pd.DataFrame, vehicle_id: str) -> Optional[Dict[str, Any]]:
    """Pulls one asset's row out of an already-fetched fleet profile —
    prefer this over re-querying when a component already has the
    fleet-wide DataFrame in hand (e.g. fleet_map.py rendering a detail
    panel after the user clicks a marker)."""
    match = profile_df[profile_df["vehicle_id"] == vehicle_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_latest_vehicle_locations(_conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per vehicle that has issued at least one BMS command: its
    most recent latitude/longitude/timestamp/command_type. Vehicles with
    no command history yet are simply absent — callers (fleet_map.py)
    decide how to note that rather than this function inventing a
    placeholder location. Iterates per-vehicle rather than a raw SQL
    MAX(timestamp) join, matching the per-vehicle query pattern already
    used throughout ingestion/db.py and models/risk_engine.py."""
    from ingestion.db import get_all_vehicle_ids, get_commands_for_vehicle

    rows = []
    for vid in get_all_vehicle_ids(_conn):
        commands = get_commands_for_vehicle(_conn, vid)
        if not commands:
            continue
        latest = commands[-1]  # get_commands_for_vehicle orders by timestamp ascending
        rows.append({
            "vehicle_id": vid,
            "latitude": latest["latitude"],
            "longitude": latest["longitude"],
            "timestamp": latest["timestamp"],
            "command_type": latest["command_type"],
        })
    return pd.DataFrame(rows, columns=["vehicle_id", "latitude", "longitude", "timestamp", "command_type"])


# ----------------------------------------------------------------------
# Fleet-wide summary stats — backs app.py's top-level metric row
# ----------------------------------------------------------------------
def fleet_summary_stats(profile_df: pd.DataFrame) -> Dict[str, Any]:
    if profile_df.empty:
        return {
            "total_vehicles": 0,
            "risk_counts": {level: 0 for level in RISK_LEVEL_ORDER},
            "vehicles_needing_maintenance": 0,
            "vehicles_with_active_security_signal": 0,
            "mean_charge_stress_score": None,
        }

    risk_counts = {
        level: int((profile_df["overall_risk_level"] == level).sum())
        for level in RISK_LEVEL_ORDER
    }
    needs_maintenance = int(profile_df["status"].isin(["degraded", "critical"]).sum())
    has_security_signal = int((profile_df["max_security_severity"] != "none").sum())
    mean_stress = profile_df["charge_stress_score"].mean()

    return {
        "total_vehicles": int(len(profile_df)),
        "risk_counts": risk_counts,
        "vehicles_needing_maintenance": needs_maintenance,
        "vehicles_with_active_security_signal": has_security_signal,
        "mean_charge_stress_score": round(float(mean_stress), 1) if pd.notnull(mean_stress) else None,
    }


# ----------------------------------------------------------------------
# Formatting helpers — every "how does this number look on screen"
# decision lives here, so components stay declarative.
# ----------------------------------------------------------------------
def format_pct(value: Optional[float], decimals: int = 1, placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:.{decimals}f}%"


def format_cycles(value: Optional[float], placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:,.0f} cycles"


def format_temperature(value: Optional[float], placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:.1f}°C"


def _pluralize(noun: str, count: int) -> str:
    if count == 1:
        return noun
    if noun.endswith("y") and (len(noun) < 2 or noun[-2].lower() not in "aeiou"):
        return noun[:-1] + "ies"
    return noun + "s"


def format_count(value: Optional[int], noun: str, placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    value = int(value)
    return f"{value} {_pluralize(noun, value)}"


def format_relative_time(timestamp: Any, placeholder: str = "—") -> str:
    """'3h ago' / '2d ago' style relative formatting for the alert feed,
    so the most urgent-looking items don't require the viewer to parse
    an ISO timestamp mid-demo."""
    if timestamp is None or (isinstance(timestamp, float) and pd.isnull(timestamp)):
        return placeholder

    if isinstance(timestamp, str):
        try:
            ts = pd.to_datetime(timestamp)
        except (ValueError, TypeError):
            return placeholder
    else:
        ts = pd.Timestamp(timestamp)

    if ts.tzinfo is not None:
        now = pd.Timestamp.now(tz=ts.tzinfo)
    else:
        now = pd.Timestamp.now()

    delta = now - ts
    seconds = delta.total_seconds()

    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def badge_color(value: Optional[str], color_map: Dict[str, str], default: str = "#757575") -> str:
    """Shared lookup used by every *_COLORS map above — a single place
    that decides what an unrecognized/missing value renders as, instead
    of each component guessing its own fallback color."""
    if value is None:
        return default
    return color_map.get(value, default)


def risk_level_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    return value.replace("_", " ").title()


def status_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    return value.replace("_", " ").title()


def action_type_label(value: Optional[str]) -> str:
    return ACTION_TYPE_LABELS.get(value, value.replace("_", " ").title() if value else "Unknown")