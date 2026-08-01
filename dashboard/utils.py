from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

DEFAULT_CACHE_TTL_SECONDS = 8

# Color / label mappings
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

STATUS_PLAIN_LABELS: Dict[str, str] = {
    "healthy": "HEALTHY",
    "watch": "KEEP AN EYE",
    "degraded": "CHECKUP NEEDED",
    "critical": "NEEDS ATTENTION",
    "insufficient_data": "INSUFFICIENT DATA",
    "no_fit": "NOT ENOUGH DATA",
}

SECURITY_PLAIN_LABELS: Dict[str, str] = {
    "none": "NO SUSPICIOUS ACTIVITY",
    "low": "MINOR IRREGULARITY",
    "medium": "SUSPICIOUS COMMAND",
    "high": "UNAUTHORIZED TAMPERING",
}


def status_plain_label(value: Optional[str]) -> str:
    if not value:
        return "Status unknown"
    return STATUS_PLAIN_LABELS.get(value, "Status unknown")


def security_plain_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    return SECURITY_PLAIN_LABELS.get(value, "Unknown")

@st.cache_resource
def get_connection() -> sqlite3.Connection:
    from ingestion.db import get_connection as _get_connection, init_db

    conn = _get_connection()
    init_db(conn)
    return conn


def ensure_seeded(_conn: sqlite3.Connection) -> None:
    from ingestion.db import row_counts, DB_PATH

    if row_counts(_conn)["telemetry"] == 0:
        from scripts.seed_db import seed as _seed_db
        _seed_db(fleet_size=50, num_cycles=500, random_seed=42, db_path=DB_PATH, wipe=False)
        clear_all_caches()  # force every @st.cache_data query to reread the freshly seeded rows


def clear_all_caches() -> None:
    st.cache_data.clear()

@st.cache_resource
def get_weather_client():
    from models.weather_client import WeatherClient

    return WeatherClient()

@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_ids(_conn: sqlite3.Connection) -> List[str]:
    from ingestion.db import get_all_vehicle_ids

    return get_all_vehicle_ids(_conn)


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_fleet_profile(_conn: sqlite3.Connection) -> pd.DataFrame:
    from models.risk_engine import RiskEngine

    engine = RiskEngine()
    return engine.build_fleet_profile(_conn)


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_driver_charging_profile(_conn: sqlite3.Connection) -> pd.DataFrame:
    from models.charging_analyzer import ChargingAnalyzer

    analyzer = ChargingAnalyzer()
    return analyzer.analyze_fleet_drivers(_conn)


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_recent_actions(_conn: sqlite3.Connection, exclude_no_action: bool = True) -> pd.DataFrame:
    from agent.actions import get_all_actions

    rows = get_all_actions(_conn, exclude_no_action=exclude_no_action)
    if not rows:
        return pd.DataFrame(columns=[
            "action_id", "vehicle_id", "action_type", "priority",
            "rationale", "parameters", "status", "created_at",
        ])
    return pd.DataFrame([dict(r) for r in rows])

@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_all_vehicle_metadata(_conn: sqlite3.Connection) -> pd.DataFrame:
    from ingestion.db import get_all_vehicle_metadata as _get_all_vehicle_metadata

    rows = _get_all_vehicle_metadata(_conn)
    if not rows:
        return pd.DataFrame(columns=[
            "vehicle_id", "make", "model", "vin", "purchase_date", "warranty_expiry_date",
        ])
    return pd.DataFrame([dict(r) for r in rows])

@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_fleet_reliability_profile(_conn: sqlite3.Connection) -> pd.DataFrame:
    from models.reliability_metrics import ReliabilityAnalyzer

    return ReliabilityAnalyzer().analyze_fleet(_conn)

@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_drivers(_conn: sqlite3.Connection) -> pd.DataFrame:
    from ingestion.db import get_all_drivers

    rows = get_all_drivers(_conn)
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_current_drivers_by_vehicle(_conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    from ingestion.db import get_all_vehicle_ids, get_current_driver_for_vehicle

    result: Dict[str, Dict[str, Any]] = {}
    for vid in get_all_vehicle_ids(_conn):
        driver = get_current_driver_for_vehicle(_conn, vid)
        if driver is not None:
            result[vid] = driver
    return result

@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_fleet_range_estimates(_conn: sqlite3.Connection) -> pd.DataFrame:
    from models.range_estimator import RangeEstimator
    from simulator.config import default_config

    estimator = RangeEstimator(
        kwh_per_km=default_config.avg_kwh_per_km,
        low_range_threshold_km=default_config.low_range_threshold_km,
        low_soc_threshold_pct=default_config.low_soc_threshold_pct,
    )
    return estimator.estimate_fleet(_conn, weather_client=get_weather_client())


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_vehicle_range_estimate(_conn: sqlite3.Connection, vehicle_id: str) -> Dict[str, Any]:
    from models.range_estimator import RangeEstimator
    from simulator.config import default_config

    estimator = RangeEstimator(
        kwh_per_km=default_config.avg_kwh_per_km,
        low_range_threshold_km=default_config.low_range_threshold_km,
        low_soc_threshold_pct=default_config.low_soc_threshold_pct,
    )
    return estimator.estimate_vehicle_live(
        _conn, vehicle_id, weather_client=get_weather_client()
    ).to_dict()

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
def get_vehicle_assignments(_conn: sqlite3.Connection, vehicle_id: str) -> pd.DataFrame:
    from ingestion.db import get_assignments_for_vehicle

    rows = get_assignments_for_vehicle(_conn, vehicle_id)
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
    match = profile_df[profile_df["vehicle_id"] == vehicle_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_latest_vehicle_locations(_conn: sqlite3.Connection) -> pd.DataFrame:
    from ingestion.db import get_all_vehicle_ids, get_commands_for_vehicle

    rows = []
    for vid in get_all_vehicle_ids(_conn):
        commands = get_commands_for_vehicle(_conn, vid)
        if not commands:
            continue
        latest = commands[-1]
        rows.append({
            "vehicle_id": vid,
            "latitude": latest["latitude"],
            "longitude": latest["longitude"],
            "timestamp": latest["timestamp"],
            "command_type": latest["command_type"],
        })
    return pd.DataFrame(rows, columns=["vehicle_id", "latitude", "longitude", "timestamp", "command_type"])


@st.cache_data(ttl=DEFAULT_CACHE_TTL_SECONDS)
def get_fleet_live_state(_conn: sqlite3.Connection) -> pd.DataFrame:
    from ingestion.db import get_all_vehicle_live_state

    rows = get_all_vehicle_live_state(_conn)
    if not rows:
        return pd.DataFrame(columns=["vehicle_id", "latitude", "longitude", "activity_status", "last_moved_at"])
    return pd.DataFrame([
        {
            "vehicle_id": r["vehicle_id"], "latitude": r["latitude"], "longitude": r["longitude"],
            "activity_status": r["status"], "last_moved_at": r["last_moved_at"],
        }
        for r in rows
    ])

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

def format_pct(value: Optional[float], decimals: int = 1, placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:.{decimals}f}%"


def format_cycles(value: Optional[float], placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:,.0f} cycles"


def format_hours(value: Optional[float], placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:,.1f} hrs"


def format_km(value: Optional[float], placeholder: str = "—") -> str:
    if value is None or pd.isnull(value):
        return placeholder
    return f"{value:,.1f} km"


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