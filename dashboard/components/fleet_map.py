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

Table columns use layman-friendly labels (e.g. "Battery Status" instead
of "RUL Status") since the primary audience — fleet managers — isn't
expected to know APM/battery-engineering jargon. The map also honours
st.session_state["map_focus_vehicle"], set by attack_trigger.py right
after an attack is injected, so the map automatically zooms to the
affected vehicle instead of requiring the manager to find it manually.
The same session key is used to highlight that vehicle's row in the
fleet table in red, so it's visually unmistakable in both places.

Driver column (Future Roadmap Feature 1): build_fleet_table_view()
takes an optional driver_by_vehicle lookup (dashboard/utils.py's
get_current_drivers_by_vehicle()) and surfaces the vehicle's current
driver's name as a plain "Driver" column, right after "Vehicle" — no
new dashboard tab or picker yet, since the roadmap's own proposed shape
for this feature is exactly "a Driver column in the sortable fleet
view." A vehicle with no assignment history yet (e.g. an older seeded
DB) shows "Unassigned" rather than blank/None, matching this file's
existing "always show something explainable, never a bare null" style.

Live SoC / range (Future Roadmap Feature 2): build_fleet_table_view()
also takes an optional range_df (dashboard/utils.py's
get_fleet_range_estimates()) and surfaces "SoC" / "Est. Range" columns.
build_vehicle_scatter_data() additionally uses range_df to draw a
distinct red RING around any vehicle currently at risk of stranding —
kept as a separate visual channel from the existing risk-level FILL
color, so a vehicle can be flagged "high risk" and/or "stranding risk"
without one signal masking the other. render_fleet_summary_metrics()
adds a fifth "Stranding Risk" tile alongside the existing four. All
three degrade gracefully (fall back to "—" / no ring / a 0 count) when
range_df is omitted, matching driver_by_vehicle's existing fallback
convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.exports import export_dataframe_to_excel, export_dataframe_to_pdf
from dashboard.utils import (
    RISK_LEVEL_COLORS,
    RISK_LEVEL_ORDER,
    RUL_STATUS_COLORS,
    SECURITY_SEVERITY_COLORS,
    fleet_summary_stats,
    format_count,
    format_cycles,
    format_km,
    format_pct,
    get_current_drivers_by_vehicle,
    get_fleet_live_state,
    get_fleet_profile,
    get_fleet_range_estimates,
    get_latest_vehicle_locations,
    get_vehicle_ids,
    risk_level_label,
    security_plain_label,
    status_label,
    status_plain_label,
)
from models.anomaly_detector import DEFAULT_DEPOT_LOCATIONS

DEPOT_MARKER_COLOR = [66, 66, 66, 200]       # dark gray, for depot reference points
DEPOT_LABELS = ("Bengaluru Depot", "Delhi Depot", "Mumbai Depot")  # order matches DEFAULT_DEPOT_LOCATIONS

ATTACKED_ROW_STYLE = "background-color: #4A1420; color: #FF8A80; font-weight: 700;"

UNASSIGNED_DRIVER_LABEL = "Unassigned"

# Live SoC / range (Future Roadmap Feature 2) — a distinct ring color/
# width from the fill color already used for overall_risk_level, so
# "high risk" and "stranding risk" stay visually independent signals.
STRANDING_RISK_RING_COLOR = [255, 82, 82, 255]   # bright red ring
NORMAL_RING_COLOR = [255, 255, 255, 255]
STRANDING_RISK_LINE_WIDTH = 3
NORMAL_LINE_WIDTH = 1

# Live vehicle state (scripts/live_feed.py) — inactive vehicles get a
# dimmer marker (lower alpha) rather than a different color, so
# overall_risk_level's fill color stays the one consistent "what color
# means what" signal across the whole dashboard.
ACTIVITY_STATUS_LABELS = {"active": "🟢 Active", "inactive": "⚪ Inactive"}
ACTIVE_MARKER_ALPHA = 200
INACTIVE_MARKER_ALPHA = 90


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> List[int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


# ----------------------------------------------------------------------
# Summary metrics + risk distribution
# ----------------------------------------------------------------------
def render_fleet_summary_metrics(
    profile_df: pd.DataFrame, range_df: Optional[pd.DataFrame] = None
) -> None:
    stats = fleet_summary_stats(profile_df)
    stranding_risk_count = (
        int(range_df["at_risk_of_stranding"].sum())
        if range_df is not None and not range_df.empty
        else 0
    )

    cols = st.columns(5)
    cols[0].metric("Fleet Size", stats["total_vehicles"])
    cols[1].metric("Needs Maintenance", stats["vehicles_needing_maintenance"])
    cols[2].metric("Active Security Signals", stats["vehicles_with_active_security_signal"])
    cols[3].metric(
        "Avg Charge Stress",
        format_pct(stats["mean_charge_stress_score"], decimals=0)
        if stats["mean_charge_stress_score"] is not None else "—",
    )
    cols[4].metric("⚠️ Stranding Risk (low SoC/range)", stranding_risk_count)

    badges = " ".join(
        f"<span style='background-color:{RISK_LEVEL_COLORS[level]};color:white;"
        f"padding:3px 12px;border-radius:14px;font-size:0.8rem;font-weight:600;"
        f"margin-right:6px;box-shadow:0 1px 2px rgba(0,0,0,0.15);'>"
        f"{risk_level_label(level)}: {stats['risk_counts'][level]}</span>"
        for level in RISK_LEVEL_ORDER
    )
    st.markdown(f"<div style='margin-top:6px;'>{badges}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sortable asset table + selection
# ----------------------------------------------------------------------
def build_fleet_table_view(
    profile_df: pd.DataFrame,
    driver_by_vehicle: Optional[Dict[str, Dict[str, Any]]] = None,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pure transform: picks/renames/formats the columns worth showing
    in the fleet table, in a fixed risk-descending sort so the assets
    that most need attention are always at the top. Column labels are
    plain-English on purpose — this table is what a non-technical fleet
    manager looks at first.

    driver_by_vehicle: optional {vehicle_id: {"name": ..., ...}} lookup
    (dashboard/utils.get_current_drivers_by_vehicle()) — when omitted,
    every row's "Driver" column reads "Unassigned" rather than the
    column being dropped entirely, so callers/tests that don't pass it
    still get a stable, predictable shape.

    range_df: optional output of models.range_estimator.RangeEstimator
    .estimate_fleet() / dashboard.utils.get_fleet_range_estimates() —
    when omitted, "SoC" / "Est. Range" read "—" rather than the columns
    being dropped, matching driver_by_vehicle's fallback convention.

    live_state_df: optional dashboard.utils.get_fleet_live_state() output
    — when omitted (or a vehicle is absent from it, e.g. the live feed
    hasn't run yet), "Activity" reads "—" rather than being dropped."""
    if profile_df.empty:
        return profile_df

    driver_by_vehicle = driver_by_vehicle or {}
    range_by_vehicle = (
        range_df.set_index("vehicle_id").to_dict(orient="index")
        if range_df is not None and not range_df.empty else {}
    )
    activity_by_vehicle = (
        live_state_df.set_index("vehicle_id")["activity_status"].to_dict()
        if live_state_df is not None and not live_state_df.empty else {}
    )

    rank = {level: i for i, level in enumerate(RISK_LEVEL_ORDER)}
    df = profile_df.copy()
    df["_risk_rank"] = df["overall_risk_level"].map(rank).fillna(-1)
    df = df.sort_values(["_risk_rank", "charge_stress_score"], ascending=[False, False])

    view = pd.DataFrame({
        "Vehicle": df["vehicle_id"],
        "Driver": df["vehicle_id"].map(
            lambda vid: driver_by_vehicle.get(vid, {}).get("name") or UNASSIGNED_DRIVER_LABEL
        ),
        "Activity": df["vehicle_id"].map(
            lambda vid: ACTIVITY_STATUS_LABELS.get(activity_by_vehicle.get(vid), "—")
        ),
        "Risk": df["overall_risk_level"].map(risk_level_label),
        "Battery Status": df["status"].map(status_plain_label),
        "Est. Time Left": df["rul_cycles"].map(lambda v: format_cycles(v)),
        "SoC": df["vehicle_id"].map(
            lambda vid: format_pct(range_by_vehicle.get(vid, {}).get("soc_pct"), decimals=0)
        ),
        "Est. Range": df["vehicle_id"].map(
            lambda vid: format_km(range_by_vehicle.get(vid, {}).get("estimated_range_km"))
        ),
        "Overheating Events": df["thermal_anomaly_count"].map(lambda v: format_count(v, "event")),
        "Security": df["max_security_severity"].map(security_plain_label),
        "Charging Stress": df["charge_stress_score"].map(lambda v: format_pct(v, decimals=0)),
    })
    return view


def _style_attacked_row(row: pd.Series, attacked_vehicle_id: Optional[str]) -> List[str]:
    """Returns a per-cell style list for one row of the fleet table —
    every cell in the currently-attacked vehicle's row gets a red
    highlight, so it's impossible to miss even if the manager isn't
    looking at the map or alert feed. No-op (empty style) for every
    other row."""
    if attacked_vehicle_id and row.get("Vehicle") == attacked_vehicle_id:
        return [ATTACKED_ROW_STYLE] * len(row)
    return [""] * len(row)


def render_fleet_table(
    profile_df: pd.DataFrame,
    driver_by_vehicle: Optional[Dict[str, Dict[str, Any]]] = None,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Renders the table and a vehicle picker below it. Returns the
    selected vehicle_id (or None if the fleet is empty) — app.py uses
    this to decide which asset's detail views to render."""
    if profile_df.empty:
        st.info("No vehicles in the fleet profile yet.")
        return None

    view = build_fleet_table_view(
        profile_df, driver_by_vehicle=driver_by_vehicle, range_df=range_df, live_state_df=live_state_df
    )

    # attack_trigger.py sets this the moment a live attack is injected —
    # reused here (same key fleet_map.py's map view already reads) so the
    # table and map both point at the same vehicle without a second
    # source of truth.
    attacked_vehicle_id = st.session_state.get("map_focus_vehicle")

    styled_view = view.style.apply(_style_attacked_row, attacked_vehicle_id=attacked_vehicle_id, axis=1)
    st.dataframe(styled_view, use_container_width=True, hide_index=True)

    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "⬇️ Export Excel", data=export_dataframe_to_excel(view, sheet_name="Fleet"),
        file_name="voltsentinel_fleet_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    export_col2.download_button(
        "⬇️ Export PDF", data=export_dataframe_to_pdf(view, title="VoltSentinel Fleet Report"),
        file_name="voltsentinel_fleet_report.pdf", mime="application/pdf",
    )

    if attacked_vehicle_id and attacked_vehicle_id in view["Vehicle"].values:
        st.caption(f"🔴 **{attacked_vehicle_id}** is highlighted — unauthorized command detected.")

    vehicle_ids = view["Vehicle"].tolist()  # already risk-sorted, so the default is the most urgent asset
    return st.selectbox("Inspect asset", vehicle_ids, key="fleet_map_vehicle_select")


# ----------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------
def _merge_live_locations(locations_df: pd.DataFrame, live_state_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Prefers live_state_df's continuously-updated position/activity for
    a vehicle when present, falling back to locations_df's command-
    derived last-known position for vehicles the live feed hasn't
    touched yet (or hasn't run at all)."""
    if live_state_df is None or live_state_df.empty:
        base = locations_df.copy()
        base["activity_status"] = None
        return base

    live = live_state_df[["vehicle_id", "latitude", "longitude", "activity_status"]].copy()
    command_only = locations_df[~locations_df["vehicle_id"].isin(live["vehicle_id"])]
    if command_only.empty:
        return live
    extra = command_only[["vehicle_id", "latitude", "longitude"]].copy()
    extra["activity_status"] = None
    return pd.concat([live, extra], ignore_index=True)


def build_vehicle_scatter_data(
    locations_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merges current location with risk level (fill color) and,
    when range_df is supplied, stranding-risk status (ring color/width)
    — pure data prep, no pydeck/streamlit objects, so this is
    straightforward to unit test.

    Fill color still encodes overall_risk_level exactly as before; the
    ring around each marker is the independent stranding-risk channel.
    live_state_df (scripts/live_feed.py's output) additionally dims
    inactive vehicles' markers (lower fill alpha) rather than changing
    their color, so risk-level color stays the one consistent signal."""
    base = _merge_live_locations(locations_df, live_state_df)
    if base.empty:
        return base

    merged = base.merge(
        profile_df[["vehicle_id", "overall_risk_level", "status"]], on="vehicle_id", how="left"
    )
    merged["overall_risk_level"] = merged["overall_risk_level"].fillna("minimal")
    merged["fill_color"] = merged.apply(
        lambda r: _hex_to_rgba(
            RISK_LEVEL_COLORS.get(r["overall_risk_level"], "#757575"),
            alpha=INACTIVE_MARKER_ALPHA if r["activity_status"] == "inactive" else ACTIVE_MARKER_ALPHA,
        ),
        axis=1,
    )

    if range_df is not None and not range_df.empty:
        merged = merged.merge(
            range_df[["vehicle_id", "estimated_range_km", "soc_pct", "at_risk_of_stranding"]],
            on="vehicle_id", how="left",
        )
        merged["at_risk_of_stranding"] = merged["at_risk_of_stranding"].fillna(False)
    else:
        merged["at_risk_of_stranding"] = False
        merged["estimated_range_km"] = None
        merged["soc_pct"] = None

    merged["line_color"] = merged["at_risk_of_stranding"].map(
        lambda at_risk: STRANDING_RISK_RING_COLOR if at_risk else NORMAL_RING_COLOR
    )
    merged["line_width"] = merged["at_risk_of_stranding"].map(
        lambda at_risk: STRANDING_RISK_LINE_WIDTH if at_risk else NORMAL_LINE_WIDTH
    )
    merged["label"] = merged.apply(
        lambda r: (
            f"{r['vehicle_id']} — {risk_level_label(r['overall_risk_level'])} risk"
            + (" — ⚠️ LOW RANGE" if r["at_risk_of_stranding"] else "")
            + (f" — {ACTIVITY_STATUS_LABELS[r['activity_status']]}" if r["activity_status"] in ACTIVITY_STATUS_LABELS else "")
        ),
        axis=1,
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
        get_line_color="line_color",
        get_line_width="line_width",
        line_width_min_pixels=1,
    )


def _initial_view_state(scatter_df: pd.DataFrame) -> pdk.ViewState:
    """Zooms to the just-attacked vehicle when
    st.session_state["map_focus_vehicle"] is set (attack_trigger.py sets
    this right after injecting an attack), so the fleet manager sees the
    affected vehicle immediately instead of having to hunt for it on a
    fleet-wide view. Falls back to the normal fleet-wide average view
    otherwise."""
    focus_vehicle = st.session_state.get("map_focus_vehicle")
    if focus_vehicle and not scatter_df.empty:
        match = scatter_df[scatter_df["vehicle_id"] == focus_vehicle]
        if not match.empty:
            row = match.iloc[0]
            return pdk.ViewState(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                zoom=11,
            )

    depot_lats = [lat for lat, _ in DEFAULT_DEPOT_LOCATIONS]
    depot_lons = [lon for _, lon in DEFAULT_DEPOT_LOCATIONS]
    lats = depot_lats + (scatter_df["latitude"].tolist() if not scatter_df.empty else [])
    lons = depot_lons + (scatter_df["longitude"].tolist() if not scatter_df.empty else [])
    return pdk.ViewState(
        latitude=sum(lats) / len(lats),
        longitude=sum(lons) / len(lons),
        zoom=3.6,
    )


def render_fleet_map(
    conn, profile_df: pd.DataFrame,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> None:
    locations_df = get_latest_vehicle_locations(conn)
    all_vehicle_ids = get_vehicle_ids(conn)

    scatter_df = build_vehicle_scatter_data(locations_df, profile_df, range_df=range_df, live_state_df=live_state_df)
    missing = len(all_vehicle_ids) - len(scatter_df)

    layers = [build_depot_layer()]
    if not scatter_df.empty:
        layers.append(build_vehicle_layer(scatter_df))

    deck = pdk.Deck(
        map_style=None,  # falls back to pydeck's default light basemap, no Mapbox token required
        initial_view_state=_initial_view_state(scatter_df),
        layers=layers,
        tooltip={"text": "{label}"} if not scatter_df.empty else None,
    )
    st.pydeck_chart(deck)

    if missing > 0:
        st.caption(
            f"{format_count(missing, 'vehicle')} not shown — no BMS command or live-feed history yet "
            f"(dark gray markers are depot locations; red ring = low SoC/range; dim marker = inactive)."
        )
    else:
        st.caption("Dark gray markers are depot locations. Red ring = low SoC/range. Dim marker = inactive.")


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_fleet_overview(conn) -> Tuple[pd.DataFrame, Optional[str]]:
    """Renders summary metrics, the map, and the asset table in one
    section. Returns (profile_df, selected_vehicle_id) so app.py can
    reuse the already-fetched profile for the detail views below
    instead of re-querying it."""
    profile_df = get_fleet_profile(conn)
    driver_by_vehicle = get_current_drivers_by_vehicle(conn)
    range_df = get_fleet_range_estimates(conn)
    live_state_df = get_fleet_live_state(conn)

    render_fleet_summary_metrics(profile_df, range_df=range_df)
    st.markdown("#### 🗺️ Fleet Map")
    render_fleet_map(conn, profile_df, range_df=range_df, live_state_df=live_state_df)
    st.markdown("#### 📋 Fleet Assets")
    selected_vehicle_id = render_fleet_table(
        profile_df, driver_by_vehicle=driver_by_vehicle, range_df=range_df, live_state_df=live_state_df
    )

    return profile_df, selected_vehicle_id