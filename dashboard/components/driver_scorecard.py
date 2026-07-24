"""
dashboard/components/driver_scorecard.py

Driver Scorecard tab (Future Roadmap Feature 5 — Driver-Level
Coaching). Re-aggregates charging behaviour by driver_id instead of
vehicle_id (models/charging_analyzer.py's analyze_fleet_drivers(),
built on top of Feature 1's vehicle_assignments), so two drivers
rotating through the same vehicle are scored separately instead of
being blended into one vehicle-level signal.

Same split as every other dashboard/components/*.py file:
    - build_driver_scorecard_view(...) -- pure, no st.*.
    - render_driver_scorecard(...)     -- the only function here
      touching st.*.

Depends on Feature 1's drivers/vehicle_assignments tables already being
populated -- a DB seeded before that feature existed just shows "no
driver assignment history yet" rather than erroring.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.utils import format_pct

STRESS_TREND_LABELS = {
    "increasing": "⚠️ Worsening",
    "decreasing": "✅ Improving",
    "stable": "Stable",
    "insufficient_data": "Not enough data yet",
}


def build_driver_scorecard_view(driver_profile_df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: plain-English columns, sorted worst-charge-stress
    first so the drivers most worth a coaching conversation are at the
    top -- mirrors fleet_map.py's build_fleet_table_view() risk-first
    sort."""
    if driver_profile_df.empty:
        return driver_profile_df

    df = driver_profile_df.sort_values("charge_stress_score", ascending=False)

    def _signed_pct(v):
        if v is None or pd.isnull(v):
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.0f}%"

    return pd.DataFrame({
        "Driver": df["driver_id"],
        "Vehicles Driven": df["vehicle_count"],
        "Charge Cycles Observed": df["total_cycles"],
        "Fast-Charge Frequency": df["fast_charge_frequency_pct"].map(lambda v: format_pct(v, decimals=0)),
        "vs. Fleet Baseline": df["fast_charge_vs_baseline_pct"].map(_signed_pct),
        "Mean Depth of Discharge": df["mean_dod_pct"].map(lambda v: format_pct(v, decimals=0)),
        "Trend": df["stress_trend"].map(lambda v: STRESS_TREND_LABELS.get(v, v)),
        "Charge Stress Score": df["charge_stress_score"].map(lambda v: format_pct(v, decimals=0)),
        "Coaching Suggestion": df["suggested_policy"].fillna("No concerns currently"),
    })


def render_driver_scorecard(conn) -> None:
    from dashboard.utils import get_driver_charging_profile

    st.subheader("Driver Scorecard")
    st.caption(
        "Charging behaviour re-aggregated by driver rather than vehicle -- so a driver "
        "rotating across several vehicles is scored on their own behaviour, not blended "
        "with whoever else has driven the same vehicle."
    )

    driver_profile_df = get_driver_charging_profile(conn)
    if driver_profile_df.empty:
        st.info(
            "No driver assignment history yet -- this needs Feature 1's drivers/"
            "vehicle_assignments tables to be populated (reseed the DB if it predates "
            "that feature)."
        )
        return

    view = build_driver_scorecard_view(driver_profile_df)
    st.dataframe(view, use_container_width=True, hide_index=True)

    worst = driver_profile_df.sort_values("charge_stress_score", ascending=False).iloc[0]
    if pd.notnull(worst.get("suggested_policy")):
        st.info(
            f"Highest charge-stress driver: **{worst['driver_id']}** "
            f"({format_pct(worst['charge_stress_score'], decimals=0)} stress score) -- "
            f"{worst['suggested_policy']}"
        )