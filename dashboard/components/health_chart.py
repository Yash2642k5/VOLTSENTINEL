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

Live SoC / range (Future Roadmap Feature 2): render_health_summary_metrics
now also takes an optional range_row (dashboard/utils.py's
get_vehicle_range_estimate()) and surfaces a same-day "Live SoC / Range"
metric plus a stranding-risk warning banner — distinct from the other
four metrics here, which are all longer-horizon/aggregate signals from
risk_engine.py's profile. This mirrors fleet_map.py's fleet-wide version
of the same signal (its "Stranding Risk" tile + red map ring), just
scoped to the one asset currently open in this tab. Falls back to "—"
when range_row is omitted, matching every other optional-param pattern
already used in this file (e.g. profile_row.get(...) throughout).

Weather-aware range (Future Roadmap Feature 8): range_row now may also
carry `ambient_temp_c` / `weather_adjustment_factor` (from
models/range_estimator.py, via dashboard/utils.get_vehicle_range_estimate()).
render_health_summary_metrics appends the live ambient temperature to the
"Live SoC / Range" metric text when available, and the stranding-risk
banner notes when the range figure has been adjusted for weather. Both
are purely additive — a range_row without these keys (e.g. weather lookup
failed, or the dict came from an older/weather-agnostic call) renders
exactly as it did before this feature, since both checks use
`.get(...)` and only add text when the value is actually present.
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

    # Live SoC/range is a same-day operational signal, distinct from the
    # longer-horizon RUL/thermal metrics below — surfaced as a standalone
    # warning banner (not just a metric tile) when the vehicle is
    # currently at risk of stranding, so it's impossible to miss even if
    # the manager only glances at this tab.
    #
    # Weather-aware range (Feature 8): when the range estimate carried a
    # live ambient temperature that actually pushed the adjustment factor
    # above ~1% (i.e. weather materially affected this number, not just a
    # rounding no-op), note that in the banner so the manager understands
    # *why* the range figure might read lower than a naive kWh/rate
    # calculation would suggest.
    if range_row and range_row.get("at_risk_of_stranding"):
        weather_note = ""
        if (
            range_row.get("ambient_temp_c") is not None
            and range_row.get("weather_adjustment_factor", 1.0) > 1.01
        ):
            weather_note = f" (range adjusted for {range_row['ambient_temp_c']:.0f}°C ambient)"
        st.warning(
            f"⚠️ **Low SoC/range** — currently at {format_pct(range_row.get('soc_pct'), decimals=0)} "
            f"state of charge, an estimated {format_km(range_row.get('estimated_range_km'))} of "
            f"range remaining{weather_note}.",
            icon="⚠️",
        )

    data_quality_notes = []
    if row.get("is_stale"):
        hours = row.get("hours_since_last_reading")
        data_quality_notes.append(
            f"no telemetry in {hours:.0f}h" if _is_valid_number(hours) else "stale sensor feed"
        )
    if _is_valid_number(row.get("missing_cycle_count")) and row["missing_cycle_count"] > 0:
        data_quality_notes.append(f"{int(row['missing_cycle_count'])} missing cycle(s)")
    if _is_valid_number(row.get("out_of_range_jump_count")) and row["out_of_range_jump_count"] > 0:
        data_quality_notes.append(f"{int(row['out_of_range_jump_count'])} out-of-range reading(s)")
    if data_quality_notes:
        st.warning(f"📡 **Data quality issue** — {', '.join(data_quality_notes)}.", icon="📡")

    cols = st.columns(5)
    cols[0].metric("Current Capacity", format_pct(profile_row.get("current_capacity_pct")))
    cols[1].metric("Projected RUL", format_cycles(profile_row.get("rul_cycles")))
    cols[2].metric("Thermal Anomalies", format_count(profile_row.get("thermal_anomaly_count"), "anomaly"))
    cols[3].metric("Latest Temp", format_temperature(profile_row.get("latest_temp_c")))

    # Live SoC/Range value — kept short enough to never truncate inside
    # st.metric's fixed-width tile. The live ambient temperature (Feature
    # 8), when available, is shown as a caption underneath instead of
    # appended inline, since st.metric clips long strings rather than
    # wrapping them.
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
        # charging_analyzer.py returns None (not NaN) for "no concerns" --
        # but building the fleet profile into a pandas DataFrame silently
        # converts that None into float('nan') once it shares a column with
        # other vehicles' real string values. NaN is truthy in Python (a
        # plain `if suggested_policy:` does not catch it), so without this
        # explicit null/NaN check every "no concerns" vehicle rendered the
        # literal text "Suggested charge policy: nan" instead of nothing.
        # _is_valid_number (defined above) is really a general not-null-or-
        # NaN check despite the name -- reused here rather than duplicating
        # the same pd.isnull() logic a second time in this file.
        if _is_valid_number(suggested_policy):
            st.info(f"Suggested charge policy: {suggested_policy}")
        st.altair_chart(build_charging_chart(telemetry_df), use_container_width=True)