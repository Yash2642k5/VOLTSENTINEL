"""
dashboard/app.py

Streamlit entrypoint. Assembles every component built in Phases 3-5 into
the single unified operator view described in the project doc's §4.1
("single unified dashboard combining health, prescriptive, and security
signals") and §5.1's Presentation row. Last file in the build order —
it only wires existing pieces together; no new business logic lives here.

Layout:
    Alert banner — shown right under the header, on every tab, whenever
              attack_trigger.py has just raised a security escalation
              (st.session_state["dashboard_alert"]). This is what makes
              the notification automatic instead of requiring the
              manager to open the Alert Feed or Agent tab to find out.
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
attack_trigger.py cannot write to that widget key directly (Streamlit
forbids mutating a widget's key after it's been instantiated in the same
run), so it stashes the target vehicle under "pending_vehicle_focus"
instead; main() applies that to the real widget key here, before
render_fleet_overview() creates the selectbox for this run.

Theming: base theme (dark) is pinned in .streamlit/config.toml, so it
doesn't depend on a visitor's OS/browser setting. inject_custom_css()
below builds a genuinely dark UI on top of that — dark card backgrounds,
light text, and desaturated borders — rather than light-themed cards
(white backgrounds, dark text) sitting on a dark page, which is what
produced the half-dark/half-light look before.

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

# Load GEMINI_API_KEY (and anything else in .env.example) from a .env file
# in the project root into the real process environment, BEFORE
# agent_recommendations.py's get_decision_engine() checks os.environ for it
# further down in this same import chain. Without this, GEMINI_API_KEY has
# to be exported in the shell every session — a .env file is the whole
# point of having .env.example, but nothing was actually loading it.
# Safe no-op if python-dotenv isn't installed or no .env file exists yet.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

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
# Global styling — a genuinely dark UI (dark surfaces, light text) built
# on top of the "dark" base theme pinned in .streamlit/config.toml.
# Every surface (page, cards, sidebar, tabs, dataframe, alerts) is given
# an explicit dark background + light text pair — nothing is left to
# inherit a light-theme default, which is what caused the half-dark/
# half-light look previously.
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
def render_header() -> None:
    st.title("\U0001F50B VoltSentinel")
    st.caption(
        "AI-Powered EV Battery Asset Performance & Security Intelligence Agent — "
        "ET AI Hackathon 2026, Problem Statement 3 (EV Asset Performance Management Agent)"
    )
    st.markdown(f"<hr style='margin-top:0.4rem;margin-bottom:1.2rem;border-color:{DARK_BORDER};'>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Automatic attack notification banner — visible on every tab, no click
# required. Set by attack_trigger.py right after a live attack is
# injected and successfully flagged by the rule-based detector. Cleared
# either by the Dismiss button here, or by a manual "Refresh data" click
# in the sidebar (see render_sidebar) — both paths clear the SAME two
# keys (dashboard_alert + map_focus_vehicle) together, so the banner and
# the highlighted map/table row never fall out of sync with each other.
# ----------------------------------------------------------------------
def render_dashboard_alert_banner() -> None:
    alert = st.session_state.get("dashboard_alert")
    if not alert:
        return

    st.error(
        f"🚨 **Unauthorized battery command detected on {alert['vehicle_id']}.** "
        f"{alert['rationale']}",
        icon="🚨",
    )
    col1, col2 = st.columns([1, 5])
    if col1.button("Dismiss", key="dismiss_dashboard_alert"):
        st.session_state.pop("dashboard_alert", None)
        st.session_state.pop("map_focus_vehicle", None)
        st.rerun()


# ----------------------------------------------------------------------
# Sidebar — always-available controls, independent of the active tab
# ----------------------------------------------------------------------
def render_sidebar(conn, default_vehicle_id) -> None:
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🔄 Refresh data", key="sidebar_refresh"):
            # Clear BOTH keys together — clearing only map_focus_vehicle
            # would remove the red highlight from the map/table but leave
            # the banner (driven by dashboard_alert) showing, so the two
            # would fall out of sync on a plain refresh.
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    # Apply any pending post-attack vehicle focus BEFORE fleet_map.py's
    # selectbox (key "fleet_map_vehicle_select") is instantiated this run —
    # setting a widget's key after it's already rendered raises
    # StreamlitAPIException, so this has to happen first, not in the sidebar.
    if "pending_vehicle_focus" in st.session_state:
        st.session_state["fleet_map_vehicle_select"] = st.session_state.pop("pending_vehicle_focus")

    inject_custom_css()
    render_header()
    render_dashboard_alert_banner()
    conn = get_connection()

    tab_fleet, tab_asset, tab_agent, tab_alerts = st.tabs(
        ["🚗 Fleet Overview", "🔋 Asset Detail", "🤖 Agent", "🔔 Alert Feed"]
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