from __future__ import annotations

from typing import Any, Dict, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils import (
    RUL_STATUS_COLORS,
    format_count,
    format_cycles,
    format_km,
    format_pct,
    format_temperature,
    get_vehicle_range_estimate,
    get_vehicle_row,
    get_vehicle_telemetry,
    status_label,
)
from models.anomaly_detector import (
    DEFAULT_CRITICAL_TEMP_C,
    DEFAULT_SAFE_TEMP_C,
    AnomalyDetector,
)
from models.charging_analyzer import DEFAULT_HIGH_DOD_THRESHOLD_PCT
from models.rul_model import _exp_decay

FAST_CHARGE_COLOR = "#1E88E5"
NORMAL_CHARGE_COLOR = "#9E9D24"
THERMAL_ANOMALY_COLOR = "#C62828"


def _is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and pd.isnull(value):
        return False
    return True

def build_capacity_fade_chart(telemetry_df: pd.DataFrame, profile_row: Dict[str, Any]) -> alt.Chart:
    status = profile_row.get("status", "unknown")
    line_color = RUL_STATUS_COLORS.get(status, "#1976D2")

    actual = telemetry_df[["cycle", "capacity_pct_of_rated"]].copy()
    actual_chart = (
        alt.Chart(actual)
        .mark_line(point=alt.OverlayMarkDef(size=25), color=line_color)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("capacity_pct_of_rated:Q", title="Capacity (% of rated)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("capacity_pct_of_rated:Q", title="Capacity %", format=".2f"),
            ],
        )
    )
    layers = [actual_chart]

    fitted_a = profile_row.get("fitted_a")
    decay_rate = profile_row.get("fitted_decay_rate")
    if _is_valid_number(fitted_a) and _is_valid_number(decay_rate):
        current_cycle = float(actual["cycle"].max())
        eol_cycle = profile_row.get("eol_cycle")
        x_max = float(eol_cycle) if _is_valid_number(eol_cycle) else current_cycle * 1.15
        x_max = max(x_max, current_cycle * 1.05)  # always extend a little past the last real point

        x_range = np.linspace(float(actual["cycle"].min()), x_max, 100)
        fit_df = pd.DataFrame({
            "cycle": x_range,
            "capacity_pct_of_rated": _exp_decay(x_range, float(fitted_a), float(decay_rate)),
        })
        fitted_chart = (
            alt.Chart(fit_df)
            .mark_line(strokeDash=[5, 4], color="#616161")
            .encode(x="cycle:Q", y="capacity_pct_of_rated:Q")
        )
        layers.append(fitted_chart)

        if _is_valid_number(eol_cycle):
            eol_vline = (
                alt.Chart(pd.DataFrame({"cycle": [float(eol_cycle)]}))
                .mark_rule(color="#C62828", strokeDash=[2, 2])
                .encode(x="cycle:Q")
            )
            layers.append(eol_vline)

    eol_pct = profile_row.get("end_of_life_capacity_pct")
    if _is_valid_number(eol_pct):
        eol_hline = (
            alt.Chart(pd.DataFrame({"capacity_pct_of_rated": [float(eol_pct)]}))
            .mark_rule(color="#C62828", strokeDash=[2, 2])
            .encode(y="capacity_pct_of_rated:Q")
        )
        layers.append(eol_hline)

    return alt.layer(*layers).properties(height=320).interactive()


def build_thermal_chart(telemetry_df: pd.DataFrame) -> alt.Chart:
    detector = AnomalyDetector()
    detect_input = telemetry_df.drop(columns=["thermal_event_flag"], errors="ignore")
    result = detector.detect_thermal_anomalies(detect_input)

    base = (
        alt.Chart(result)
        .mark_line(color="#546E7A")
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("temperature_c:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("temperature_c:Q", title="Temp (°C)", format=".1f"),
            ],
        )
    )

    anomalies = result[result["thermal_anomaly"]]
    anomaly_points = (
        alt.Chart(anomalies)
        .mark_circle(size=55, color=THERMAL_ANOMALY_COLOR)
        .encode(
            x="cycle:Q",
            y="temperature_c:Q",
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("temperature_c:Q", title="Temp (°C)", format=".1f"),
            ],
        )
    )

    safe_rule = (
        alt.Chart(pd.DataFrame({"y": [DEFAULT_SAFE_TEMP_C]}))
        .mark_rule(color="#F57C00", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    critical_rule = (
        alt.Chart(pd.DataFrame({"y": [DEFAULT_CRITICAL_TEMP_C]}))
        .mark_rule(color="#C62828", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    return alt.layer(base, anomaly_points, safe_rule, critical_rule).properties(height=280).interactive()

def build_charging_chart(telemetry_df: pd.DataFrame) -> alt.Chart:
    df = telemetry_df.copy()
    df["charge_type"] = np.where(df["is_fast_charge"], "Fast charge", "Standard charge")

    points = (
        alt.Chart(df)
        .mark_circle(size=40)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("dod_pct:Q", title="Depth of discharge (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "charge_type:N",
                title="Charge type",
                scale=alt.Scale(
                    domain=["Standard charge", "Fast charge"],
                    range=[NORMAL_CHARGE_COLOR, FAST_CHARGE_COLOR],
                ),
            ),
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("dod_pct:Q", title="DoD %", format=".1f"),
                alt.Tooltip("charge_type:N", title="Type"),
            ],
        )
    )

    high_dod_rule = (
        alt.Chart(pd.DataFrame({"y": [DEFAULT_HIGH_DOD_THRESHOLD_PCT]}))
        .mark_rule(color="#C62828", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    return alt.layer(points, high_dod_rule).properties(height=280).interactive()

def render_health_summary_metrics(
    profile_row: Dict[str, Any], range_row: Optional[Dict[str, Any]] = None
) -> None:
    status = profile_row.get("status", "unknown")
    color = RUL_STATUS_COLORS.get(status, "#757575")

    st.markdown(
        f"<span style='background-color:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{status_label(status)}</span>",
        unsafe_allow_html=True,
    )

    if range_row and range_row.get("at_risk_of_stranding"):
        weather_note = ""
        if (
            range_row.get("ambient_temp_c") is not None
            and range_row.get("weather_adjustment_factor", 1.0) > 1.01
        ):
            weather_note = f" (range adjusted for {range_row['ambient_temp_c']:.0f}°C ambient)"
        st.warning(
            f"**Low SoC/range** — currently at {format_pct(range_row.get('soc_pct'), decimals=0)} "
            f"state of charge, an estimated {format_km(range_row.get('estimated_range_km'))} of "
            f"range remaining{weather_note}.",
            icon="⚠️",
        )

    data_quality_notes = []
    if profile_row.get("is_stale"):
        hours = profile_row.get("hours_since_last_reading")
        data_quality_notes.append(
            f"no telemetry in {hours:.0f}h" if _is_valid_number(hours) else "stale sensor feed"
        )
    if _is_valid_number(profile_row.get("missing_cycle_count")) and profile_row["missing_cycle_count"] > 0:
        data_quality_notes.append(f"{int(profile_row['missing_cycle_count'])} missing cycle(s)")
    if _is_valid_number(profile_row.get("out_of_range_jump_count")) and profile_row["out_of_range_jump_count"] > 0:
        data_quality_notes.append(f"{int(profile_row['out_of_range_jump_count'])} out-of-range reading(s)")
    if data_quality_notes:
        st.warning(f"📡 **Data quality issue** — {', '.join(data_quality_notes)}.", icon="📡")

    cols = st.columns(5)
    cols[0].metric("Current Capacity", format_pct(profile_row.get("current_capacity_pct")))
    cols[1].metric("Projected RUL", format_cycles(profile_row.get("rul_cycles")))
    cols[2].metric("Thermal Anomalies", format_count(profile_row.get("thermal_anomaly_count"), "anomaly"))
    cols[3].metric("Latest Temp", format_temperature(profile_row.get("latest_temp_c")))

    if range_row:
        live_value = (
            f"{format_pct(range_row.get('soc_pct'), decimals=0)} · "
            f"{format_km(range_row.get('estimated_range_km'))}"
        )
    else:
        live_value = "—"
    cols[4].metric("Live SoC / Range", live_value)
    if range_row and range_row.get("ambient_temp_c") is not None:
        cols[4].caption(f"🌤️ {range_row['ambient_temp_c']:.0f}°C ambient")


def render_health_chart(conn, vehicle_id: str, profile_df: pd.DataFrame) -> None:
    row = get_vehicle_row(profile_df, vehicle_id)
    if row is None:
        st.warning(f"No risk profile available for {vehicle_id} yet.")
        return

    telemetry_df = get_vehicle_telemetry(conn, vehicle_id)
    if telemetry_df.empty:
        st.warning(f"No telemetry recorded for {vehicle_id} yet.")
        return

    range_row = get_vehicle_range_estimate(conn, vehicle_id)
    render_health_summary_metrics(row, range_row=range_row)

    tab_rul, tab_thermal, tab_charging = st.tabs(["Capacity & RUL", "Thermal", "Charging Behaviour"])

    with tab_rul:
        if row.get("status") in ("insufficient_data", "no_fit"):
            st.info(
                "Not enough telemetry yet for a reliable RUL fit — showing raw capacity readings only."
            )
        st.altair_chart(build_capacity_fade_chart(telemetry_df, row), use_container_width=True)

    with tab_thermal:
        st.altair_chart(build_thermal_chart(telemetry_df), use_container_width=True)

    with tab_charging:
        suggested_policy = row.get("suggested_policy")
        if _is_valid_number(suggested_policy):
            st.info(f"Suggested charge policy: {suggested_policy}")
        st.altair_chart(build_charging_chart(telemetry_df), use_container_width=True)