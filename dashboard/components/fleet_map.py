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

STRANDING_RISK_RING_COLOR = [255, 82, 82, 255]   # bright red ring
NORMAL_RING_COLOR = [255, 255, 255, 255]
STRANDING_RISK_LINE_WIDTH = 3
NORMAL_LINE_WIDTH = 1

ACTIVITY_STATUS_LABELS = {"active": "🟢 ACTIVE", "inactive": "⚪ INACTIVE"}
ACTIVE_MARKER_ALPHA = 200
INACTIVE_MARKER_ALPHA = 90


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> List[int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]

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

def build_fleet_table_view(
    profile_df: pd.DataFrame,
    driver_by_vehicle: Optional[Dict[str, Dict[str, Any]]] = None,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
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
    if attacked_vehicle_id and row.get("Vehicle") == attacked_vehicle_id:
        return [ATTACKED_ROW_STYLE] * len(row)
    return [""] * len(row)


def render_fleet_table(
    profile_df: pd.DataFrame,
    driver_by_vehicle: Optional[Dict[str, Dict[str, Any]]] = None,
    range_df: Optional[pd.DataFrame] = None,
    live_state_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    if profile_df.empty:
        st.info("No vehicles in the fleet profile yet.")
        return None

    view = build_fleet_table_view(
        profile_df, driver_by_vehicle=driver_by_vehicle, range_df=range_df, live_state_df=live_state_df
    )

    attacked_vehicle_id = st.session_state.get("map_focus_vehicle")

    styled_view = view.style.apply(_style_attacked_row, attacked_vehicle_id=attacked_vehicle_id, axis=1)
    st.dataframe(styled_view, use_container_width=True, hide_index=True)

    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "EXPORT EXCEL", data=export_dataframe_to_excel(view, sheet_name="Fleet"),
        file_name="voltsentinel_fleet_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    export_col2.download_button(
        "EXPORT PDF", data=export_dataframe_to_pdf(view, title="VoltSentinel Fleet Report"),
        file_name="voltsentinel_fleet_report.pdf", mime="application/pdf",
    )

    if attacked_vehicle_id and attacked_vehicle_id in view["Vehicle"].values:
        st.caption(f"**{attacked_vehicle_id}** is highlighted — unauthorized command detected.")

    vehicle_ids = view["Vehicle"].tolist()  # already risk-sorted, so the default is the most urgent asset
    return st.selectbox("Inspect asset", vehicle_ids, key="fleet_map_vehicle_select")

def _merge_live_locations(locations_df: pd.DataFrame, live_state_df: Optional[pd.DataFrame]) -> pd.DataFrame:
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

def render_fleet_overview(conn) -> Tuple[pd.DataFrame, Optional[str]]:
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