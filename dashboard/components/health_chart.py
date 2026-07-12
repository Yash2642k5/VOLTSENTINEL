"""
dashboard/components/health_chart.py

Per-asset health visualization: capacity-fade / RUL projection, thermal
trend, and charging-behaviour trend. Consumes models/ output by way of
dashboard/utils.py's cached queries — this file does not fit RUL curves
or run anomaly detection itself except where a chart genuinely needs a
per-cycle signal that risk_engine's aggregate profile doesn't expose
(thermal anomaly flags per point, so we can highlight the exact cycles).

Split in two layers, deliberately:
  - build_*_chart(...)   -> returns an altair Chart object. Pure — no
    Streamlit calls, so these are unit-testable without a running app
    (assert on encoding/data, not pixels).
  - render_health_chart(...) -> the actual page section: pulls data via
    dashboard/utils.py, calls the build_* functions, and is the only
    place in this file that touches `st.*`.

Uses Altair (bundled with Streamlit — no extra dependency) rather than
Plotly, specifically so requirements.txt doesn't grow for this.
"""

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
    format_pct,
    format_temperature,
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
from models.rul_model import _exp_decay  # reuse the exact fit formula, not a re-derivation

FAST_CHARGE_COLOR = "#1E88E5"
NORMAL_CHARGE_COLOR = "#9E9D24"
THERMAL_ANOMALY_COLOR = "#C62828"


def _is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and pd.isnull(value):
        return False
    return True


# ----------------------------------------------------------------------
# Capacity fade / RUL projection
# ----------------------------------------------------------------------
def build_capacity_fade_chart(telemetry_df: pd.DataFrame, profile_row: Dict[str, Any]) -> alt.Chart:
    """Actual capacity_pct_of_rated per cycle, plus (when the fit
    converged) the fitted decay curve extended out to the projected
    EOL cycle, and a threshold rule at end_of_life_capacity_pct."""
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


# ----------------------------------------------------------------------
# Thermal trend
# ----------------------------------------------------------------------
def build_thermal_chart(telemetry_df: pd.DataFrame) -> alt.Chart:
    """Temperature per cycle, with safe/critical threshold rules and the
    exact cycles the rule-based thermal detector flags highlighted —
    recomputed here (cheap, no ML fit involved) rather than pulled from
    risk_engine's profile, since the profile only carries the aggregate
    count, not which specific cycles triggered it."""
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


# ----------------------------------------------------------------------
# Charging behaviour
# ----------------------------------------------------------------------
def build_charging_chart(telemetry_df: pd.DataFrame) -> alt.Chart:
    """Depth-of-discharge per cycle, colored by whether that cycle was a
    fast-charge event, with a rule at the high-DoD ('abusive') threshold
    so the driver-coaching case from charging_analyzer's suggested_policy
    is visible directly on the chart, not just in the summary metric."""
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


# ----------------------------------------------------------------------
# Summary metrics row
# ----------------------------------------------------------------------
def render_health_summary_metrics(profile_row: Dict[str, Any]) -> None:
    status = profile_row.get("status", "unknown")
    color = RUL_STATUS_COLORS.get(status, "#757575")

    st.markdown(
        f"<span style='background-color:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{status_label(status)}</span>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Current Capacity", format_pct(profile_row.get("current_capacity_pct")))
    cols[1].metric("Projected RUL", format_cycles(profile_row.get("rul_cycles")))
    cols[2].metric("Thermal Anomalies", format_count(profile_row.get("thermal_anomaly_count"), "anomaly"))
    cols[3].metric("Latest Temp", format_temperature(profile_row.get("latest_temp_c")))


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_health_chart(conn, vehicle_id: str, profile_df: pd.DataFrame) -> None:
    row = get_vehicle_row(profile_df, vehicle_id)
    if row is None:
        st.warning(f"No risk profile available for {vehicle_id} yet.")
        return

    telemetry_df = get_vehicle_telemetry(conn, vehicle_id)
    if telemetry_df.empty:
        st.warning(f"No telemetry recorded for {vehicle_id} yet.")
        return

    render_health_summary_metrics(row)

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
        if suggested_policy:
            st.info(f"Suggested charge policy: {suggested_policy}")
        st.altair_chart(build_charging_chart(telemetry_df), use_container_width=True)