"""
Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
from dashboard.utils import clear_all_caches, ensure_seeded, get_connection, get_vehicle_row
from dashboard.components.agent_recommendations import render_agent_recommendations
from dashboard.components.alert_feed import render_alert_feed
from dashboard.components.asset_registry import render_asset_registry
from dashboard.components.attack_trigger import render_attack_trigger
from dashboard.components.bi_chat import render_bi_chat
from dashboard.components.driver_scorecard import render_driver_scorecard
from dashboard.components.fleet_map import render_fleet_overview
from dashboard.components.health_chart import render_health_chart


st.set_page_config(
    page_title="VoltSentinel — EV Battery APM Agent",
    page_icon="\U0001F50B",
    layout="wide",
)

DARK_BG = "#0E1420"
DARK_SURFACE = "#161E2E"
DARK_SURFACE_ALT = "#1C2536"
DARK_BORDER = "#2A3548"
TEXT_PRIMARY = "#E8ECF1"
TEXT_MUTED = "#8CA0B8"
ACCENT_BLUE = "#42A5F5"
ACCENT_TEAL = "#26C6C1"


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        /* Page background */
        .stApp {{
            background: linear-gradient(180deg, {DARK_BG} 0%, #0A0F18 100%) !important;
        }}
        .main .block-container {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Header title */
        h1 {{
            font-weight: 800 !important;
            background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_TEAL});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}

        /* All headings + body text in markdown containers */
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] span,
        div[data-testid="stMarkdownContainer"] li {{
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] h4 {{
            font-weight: 700 !important;
        }}
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] * {{
            color: {TEXT_MUTED} !important;
        }}

        hr {{
            border-color: {DARK_BORDER} !important;
        }}

        /* Metric cards — dark surface, light values */
        div[data-testid="stMetric"] {{
            background: {DARK_SURFACE} !important;
            border: 1px solid {DARK_BORDER};
            border-radius: 14px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        }}
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {{
            font-weight: 600 !important;
            color: {TEXT_MUTED} !important;
            opacity: 1 !important;
        }}
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {{
            font-weight: 800 !important;
            color: {ACCENT_BLUE} !important;
            opacity: 1 !important;
        }}

        /* Tabs */
        div[data-baseweb="tab-list"] {{
            border-bottom: 1px solid {DARK_BORDER} !important;
            gap: 4px;
        }}
        button[data-baseweb="tab"] {{
            font-weight: 600;
            font-size: 0.95rem;
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            background: transparent !important;
        }}
        button[data-baseweb="tab"] p {{
            color: {TEXT_MUTED} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_TEAL}) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: #0A0F18 !important;
            font-weight: 700 !important;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid {DARK_BORDER};
            background: {DARK_SURFACE_ALT} !important;
            color: {TEXT_PRIMARY} !important;
            transition: transform 0.05s ease-in-out;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.45);
            border-color: {ACCENT_BLUE};
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, #D32F2F, #C2185B) !important;
            border: none !important;
            color: #FFFFFF !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: #0A0F18 !important;
            border-right: 1px solid {DARK_BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT_PRIMARY} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] * {{
            color: {TEXT_MUTED} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: {DARK_SURFACE} !important;
            border: 1px solid {DARK_BORDER};
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: {DARK_SURFACE_ALT} !important;
            border-color: {ACCENT_BLUE};
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: {DARK_SURFACE} !important;
            border-color: {DARK_BORDER} !important;
            color: {TEXT_PRIMARY} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: {DARK_BORDER} !important;
        }}

        /* Dropdown menus / listboxes (selectbox popovers) */
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background: {DARK_SURFACE} !important;
        }}
        ul[data-testid="stSelectboxVirtualDropdown"] li {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Alert / error banner */
        div[data-testid="stAlert"] {{
            border-radius: 12px;
            font-weight: 500;
        }}

        /* Dataframe */
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {DARK_BORDER};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        }}

        /* Expanders */
        div[data-testid="stExpander"] {{
            border-radius: 10px;
            border: 1px solid {DARK_BORDER};
            background: {DARK_SURFACE} !important;
        }}
        div[data-testid="stExpander"] summary {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Info boxes */
        div[data-baseweb="notification"] {{
            background: {DARK_SURFACE} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header() -> None:
    st.title("\U0001F50B Welcome Fleet Manager (id: 1234)")
    st.caption(
        "AI-Powered EV Battery Asset Performance & Security Intelligence Agent — VOLTSENTINEL"
    )
    st.markdown(f"<hr style='margin-top:0.4rem;margin-bottom:1.2rem;border-color:{DARK_BORDER};'>",
                unsafe_allow_html=True)

def render_dashboard_alert_banner() -> None:
    alert = st.session_state.get("dashboard_alert")
    if not alert:
        return

    st.error(
        f"**Unauthorized battery command detected on {alert['vehicle_id']}.** "
        f"{alert['rationale']}",
        icon="🚨",
    )
    col1, col2 = st.columns([1, 5])
    if col1.button("Dismiss", key="dismiss_dashboard_alert"):
        st.session_state.pop("dashboard_alert", None)
        st.session_state.pop("map_focus_vehicle", None)
        st.rerun()

def render_sidebar(conn, default_vehicle_id) -> None:
    with st.sidebar:
        st.header("CONTROLS")

        if st.button("Refresh data", key="sidebar_refresh"):
            st.session_state.pop("map_focus_vehicle", None)
            st.session_state.pop("dashboard_alert", None)
            clear_all_caches()
            st.rerun()

        st.caption(
            "Fleet data auto-refreshes every few seconds; use this to force an "
            "immediate reload (e.g. right after injecting an attack)."
        )

        st.divider()
        render_attack_trigger(conn, default_vehicle_id=default_vehicle_id)

def main() -> None:
    if "pending_vehicle_focus" in st.session_state:
        st.session_state["fleet_map_vehicle_select"] = st.session_state.pop("pending_vehicle_focus")

    inject_custom_css()
    render_header()
    render_dashboard_alert_banner()
    conn = get_connection()
    ensure_seeded(conn) 

    tab_fleet, tab_registry, tab_bi, tab_asset, tab_agent, tab_alerts, tab_drivers = st.tabs(
        [
            " FLEET OVERVIEW", "ASSETS", "FLEET BI", "ASSET DETAIL",
            "AGENT REASONING", "ALERT FEED", "DRIVER",
        ]
    )

    with tab_fleet:
        profile_df, selected_vehicle_id = render_fleet_overview(conn)
    selected_vehicle_id = st.session_state.get("fleet_map_vehicle_select", selected_vehicle_id)

    with tab_registry:
        render_asset_registry(conn, profile_df)

    with tab_bi:
        render_bi_chat(conn, profile_df)

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

    with tab_drivers:
        render_driver_scorecard(conn)

    render_sidebar(conn, default_vehicle_id=selected_vehicle_id)


if __name__ == "__main__":
    main()