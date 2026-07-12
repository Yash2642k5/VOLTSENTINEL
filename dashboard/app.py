"""
dashboard/app.py

Streamlit entrypoint. Assembles every component built in Phases 3-5 into
the single unified operator view described in the project doc's §4.1
("single unified dashboard combining health, prescriptive, and security
signals") and §5.1's Presentation row. Last file in the build order —
it only wires existing pieces together; no new business logic lives here.

Layout:
    Sidebar — the live "Simulate Attack" trigger (attack_trigger.py) and
            a manual refresh button, both reachable regardless of which
            tab is open, since the attack trigger is the demo's single
            most important interaction.
    Tab 1   — Fleet Overview: summary metrics, map, sortable table
            (fleet_map.py). Selecting a row here is what drives every
            other tab's "current asset".
    Tab 2   — Asset Detail: per-asset health/thermal/charging charts
            (health_chart.py) for whichever vehicle is selected.
    Tab 3   — Agent: live Perceive->Reason->Decide->Act reasoning and
            decision history for the selected asset
            (agent_recommendations.py).
    Tab 4   — Alert Feed: fleet-wide reverse-chronological ticker of
            everything the agent has actually logged (alert_feed.py).

Vehicle selection is threaded through st.session_state["fleet_map_vehicle_select"]
— the same key fleet_map.py's selectbox already writes to — rather than a
second piece of state here, so there's exactly one source of truth for
"which asset is currently open," no matter which tab last changed it.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit inserts this script's own directory (dashboard/) into sys.path,
# not the project root — so `from dashboard.components... import ...` below
# would fail with "No module named 'dashboard'" no matter what directory
# `streamlit run` is invoked from. Explicitly add the project root (this
# file's grandparent) before any dashboard.*/models.*/agent.* import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.agent_recommendations import render_agent_recommendations
from dashboard.components.alert_feed import render_alert_feed
from dashboard.components.attack_trigger import render_attack_trigger
from dashboard.components.fleet_map import render_fleet_overview
from dashboard.components.health_chart import render_health_chart
from dashboard.utils import clear_all_caches, get_connection, get_vehicle_row

st.set_page_config(
    page_title="VoltSentinel — EV Battery APM Agent",
    page_icon="\U0001F50B",
    layout="wide",
)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
def render_header() -> None:
    st.title("\U0001F50B VoltSentinel")
    st.caption(
        "AI-Powered EV Battery Asset Performance & Security Intelligence Agent — "
        "ET AI Hackathon 2026, Problem Statement 3 (EV Asset Performance Management Agent)"
    )


# ----------------------------------------------------------------------
# Sidebar — always-available controls, independent of the active tab
# ----------------------------------------------------------------------
def render_sidebar(conn, default_vehicle_id) -> None:
    with st.sidebar:
        st.header("Controls")

        if st.button("Refresh data", key="sidebar_refresh"):
            clear_all_caches()
            st.rerun()

        st.caption(
            "Fleet data auto-refreshes every few seconds; use this to force an "
            "immediate reload (e.g. right after injecting an attack)."
        )

        st.divider()
        render_attack_trigger(conn, default_vehicle_id=default_vehicle_id)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    render_header()
    conn = get_connection()

    tab_fleet, tab_asset, tab_agent, tab_alerts = st.tabs(
        ["Fleet Overview", "Asset Detail", "Agent", "Alert Feed"]
    )

    with tab_fleet:
        profile_df, selected_vehicle_id = render_fleet_overview(conn)

    # fleet_map.py's selectbox (key "fleet_map_vehicle_select") is the single
    # source of truth for the current asset — read it back here so tabs 2-4
    # and the sidebar all agree on the same selection even though only
    # tab_fleet's body actually rendered the picker widget.
    selected_vehicle_id = st.session_state.get("fleet_map_vehicle_select", selected_vehicle_id)

    with tab_asset:
        if selected_vehicle_id is None:
            st.info("No asset selected yet — pick one from the Fleet Overview tab.")
        else:
            st.subheader(f"Asset Detail — {selected_vehicle_id}")
            render_health_chart(conn, selected_vehicle_id, profile_df)

    with tab_agent:
        if selected_vehicle_id is None:
            st.info("No asset selected yet — pick one from the Fleet Overview tab.")
        else:
            profile_row = get_vehicle_row(profile_df, selected_vehicle_id)
            st.subheader(f"Agent Reasoning — {selected_vehicle_id}")
            render_agent_recommendations(conn, selected_vehicle_id, profile_row)

    with tab_alerts:
        st.subheader("Fleet-Wide Alert Feed")
        render_alert_feed(conn)

    render_sidebar(conn, default_vehicle_id=selected_vehicle_id)


if __name__ == "__main__":
    main()