"""
dashboard/components/fleet_map.py

Fleet-wide overview: top-level summary metrics, a sortable per-asset
table, and a geographic map of each vehicle's last-known position
(from its most recent BMS command) color-coded by overall_risk_level.

Same split as health_chart.py:
    - build_*(...) functions are pure (no st.* calls) so the geometry/
    color logic is testable without a running Streamlit app.
    - render_fleet_overview(...) is the page section app.py calls; it's
    the only function here that touches st.*, and it returns the
    vehicle_id the user selected so app.py can feed it straight into
    health_chart.render_health_chart() / agent_recommendations.py.

Uses pydeck for the map (bundled with Streamlit as a dependency of
st.pydeck_chart) rather than st.map's newer color= parameter, since
pydeck's ScatterplotLayer API has been stable for a long time and
doesn't depend on the caller being on a very recent Streamlit version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.utils import (
    RISK_LEVEL_COLORS,
    RISK_LEVEL_ORDER,
    RUL_STATUS_COLORS,
    SECURITY_SEVERITY_COLORS,
    fleet_summary_stats,
    format_count,
    format_cycles,
    format_pct,
    get_fleet_profile,
    get_latest_vehicle_locations,
    get_vehicle_ids,
    risk_level_label,
    status_label,
)
from models.anomaly_detector import DEFAULT_DEPOT_LOCATIONS

DEPOT_MARKER_COLOR = [66, 66, 66, 200]       # dark gray, for depot reference points
DEPOT_LABELS = ("Bengaluru Depot", "Delhi Depot", "Mumbai Depot")  # order matches DEFAULT_DEPOT_LOCATIONS


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> List[int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


# ----------------------------------------------------------------------
# Summary metrics + risk distribution
# ----------------------------------------------------------------------
def render_fleet_summary_metrics(profile_df: pd.DataFrame) -> None:
    stats = fleet_summary_stats(profile_df)

    cols = st.columns(4)
    cols[0].metric("Fleet Size", stats["total_vehicles"])
    cols[1].metric("Needs Maintenance", stats["vehicles_needing_maintenance"])
    cols[2].metric("Active Security Signals", stats["vehicles_with_active_security_signal"])
    cols[3].metric(
        "Avg Charge Stress",
        format_pct(stats["mean_charge_stress_score"], decimals=0)
        if stats["mean_charge_stress_score"] is not None else "—",
    )

    badges = " ".join(
        f"<span style='background-color:{RISK_LEVEL_COLORS[level]};color:white;"
        f"padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;"
        f"margin-right:6px;'>{risk_level_label(level)}: {stats['risk_counts'][level]}</span>"
        for level in RISK_LEVEL_ORDER
    )
    st.markdown(badges, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sortable asset table + selection
# ----------------------------------------------------------------------
def build_fleet_table_view(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: picks/renames/formats the columns worth showing
    in the fleet table, in a fixed risk-descending sort so the assets
    that most need attention are always at the top."""
    if profile_df.empty:
        return profile_df

    rank = {level: i for i, level in enumerate(RISK_LEVEL_ORDER)}
    df = profile_df.copy()
    df["_risk_rank"] = df["overall_risk_level"].map(rank).fillna(-1)
    df = df.sort_values(["_risk_rank", "charge_stress_score"], ascending=[False, False])

    view = pd.DataFrame({
        "Vehicle": df["vehicle_id"],
        "Risk": df["overall_risk_level"].map(risk_level_label),
        "RUL Status": df["status"].map(status_label),
        "Capacity": df["current_capacity_pct"].map(lambda v: format_pct(v)),
        "RUL": df["rul_cycles"].map(lambda v: format_cycles(v)),
        "Thermal Anomalies": df["thermal_anomaly_count"].map(lambda v: format_count(v, "anomaly")),
        "Security": df["max_security_severity"].map(lambda v: str(v).title() if pd.notnull(v) else "None"),
        "Charge Stress": df["charge_stress_score"].map(lambda v: format_pct(v, decimals=0)),
    })
    return view


def render_fleet_table(profile_df: pd.DataFrame) -> Optional[str]:
    """Renders the table and a vehicle picker below it. Returns the
    selected vehicle_id (or None if the fleet is empty) — app.py uses
    this to decide which asset's detail views to render."""
    if profile_df.empty:
        st.info("No vehicles in the fleet profile yet.")
        return None

    view = build_fleet_table_view(profile_df)
    st.dataframe(view, use_container_width=True, hide_index=True)

    vehicle_ids = view["Vehicle"].tolist()  # already risk-sorted, so the default is the most urgent asset
    return st.selectbox("Inspect asset", vehicle_ids, key="fleet_map_vehicle_select")


# ----------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------
def build_vehicle_scatter_data(locations_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """Merges last-known location with risk level and computes the RGBA
    fill color per point — pure data prep, no pydeck/streamlit objects,
    so this is straightforward to unit test."""
    if locations_df.empty:
        return locations_df

    merged = locations_df.merge(
        profile_df[["vehicle_id", "overall_risk_level", "status"]], on="vehicle_id", how="left"
    )
    merged["overall_risk_level"] = merged["overall_risk_level"].fillna("minimal")
    merged["fill_color"] = merged["overall_risk_level"].map(
        lambda level: _hex_to_rgba(RISK_LEVEL_COLORS.get(level, "#757575"))
    )
    merged["label"] = merged.apply(
        lambda r: f"{r['vehicle_id']} — {risk_level_label(r['overall_risk_level'])} risk", axis=1
    )
    return merged


def build_depot_layer() -> pdk.Layer:
    depot_df = pd.DataFrame({
        "name": DEPOT_LABELS,
        "latitude": [lat for lat, _ in DEFAULT_DEPOT_LOCATIONS],
        "longitude": [lon for _, lon in DEFAULT_DEPOT_LOCATIONS],
    })
    return pdk.Layer(
        "ScatterplotLayer",
        data=depot_df,
        get_position="[longitude, latitude]",
        get_fill_color=DEPOT_MARKER_COLOR,
        get_radius=3000,
        pickable=True,
    )


def build_vehicle_layer(scatter_df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "ScatterplotLayer",
        data=scatter_df,
        get_position="[longitude, latitude]",
        get_fill_color="fill_color",
        get_radius=1200,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )


def _initial_view_state(scatter_df: pd.DataFrame) -> pdk.ViewState:
    depot_lats = [lat for lat, _ in DEFAULT_DEPOT_LOCATIONS]
    depot_lons = [lon for _, lon in DEFAULT_DEPOT_LOCATIONS]
    lats = depot_lats + (scatter_df["latitude"].tolist() if not scatter_df.empty else [])
    lons = depot_lons + (scatter_df["longitude"].tolist() if not scatter_df.empty else [])
    return pdk.ViewState(
        latitude=sum(lats) / len(lats),
        longitude=sum(lons) / len(lons),
        zoom=3.6,
    )


def render_fleet_map(conn, profile_df: pd.DataFrame) -> None:
    locations_df = get_latest_vehicle_locations(conn)
    all_vehicle_ids = get_vehicle_ids(conn)
    missing = len(all_vehicle_ids) - len(locations_df)

    scatter_df = build_vehicle_scatter_data(locations_df, profile_df)

    layers = [build_depot_layer()]
    if not scatter_df.empty:
        layers.append(build_vehicle_layer(scatter_df))

    deck = pdk.Deck(
        map_style=None,  # falls back to pydeck's default light basemap, no Mapbox token required
        initial_view_state=_initial_view_state(scatter_df),
        layers=layers,
        tooltip={"text": "{label}\n{command_type} at {timestamp}"} if not scatter_df.empty else None,
    )
    st.pydeck_chart(deck)

    if missing > 0:
        st.caption(
            f"{format_count(missing, 'vehicle')} not shown — no BMS command history yet "
            f"(dark gray markers are depot locations)."
        )
    else:
        st.caption("Dark gray markers are depot locations.")


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_fleet_overview(conn) -> Tuple[pd.DataFrame, Optional[str]]:
    """Renders summary metrics, the map, and the asset table in one
    section. Returns (profile_df, selected_vehicle_id) so app.py can
    reuse the already-fetched profile for the detail views below
    instead of re-querying it."""
    profile_df = get_fleet_profile(conn)

    render_fleet_summary_metrics(profile_df)
    st.subheader("Fleet Map")
    render_fleet_map(conn, profile_df)
    st.subheader("Fleet Assets")
    selected_vehicle_id = render_fleet_table(profile_df)

    return profile_df, selected_vehicle_id