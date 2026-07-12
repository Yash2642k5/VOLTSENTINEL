"""
dashboard/components/agent_recommendations.py

Per-asset panel showing the agent decision layer's output for whichever
vehicle is currently selected — distinct from alert_feed.py's fleet-wide
ticker. Two sections:

  1. Live reasoning: a "Run agent reasoning" button that calls
     agent/decision_engine.py's Perceive->Reason->Decide loop for this
     one asset, previews the resulting actions, and only writes them to
     agent_actions (the Act step) once the user explicitly confirms —
     separating "what would the agent do" from "actually do it," since
     Streamlit reruns the whole script on every widget interaction and
     we don't want a stray rerun silently re-logging duplicate tickets.
  2. History: past actions already logged for this vehicle, reusing
     alert_feed.py's card renderer for a consistent look.

Requires GEMINI_API_KEY to be set for the live-reasoning section — if
it isn't, that section degrades to an explanatory message rather than
erroring, and the history section still works (it's pure DB read).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from agent.actions import execute_decision
from agent.decision_engine import DecisionEngine
from dashboard.components.alert_feed import render_alert_card
from dashboard.utils import clear_all_caches, get_vehicle_actions


# ----------------------------------------------------------------------
# Cached engine — one Gemini client per session, not re-created on
# every button click / rerun.
# ----------------------------------------------------------------------
@st.cache_resource
def get_decision_engine() -> Optional[DecisionEngine]:
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        return DecisionEngine.create()
    except Exception:
        # Bad key, network issue at client construction time, etc. — treat
        # the same as "unavailable" rather than crashing the dashboard.
        return None


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------
def decision_action_to_row(vehicle_id: str, action: Dict[str, Any], status: str) -> Dict[str, Any]:
    """Reshapes one action from a decision_engine result (action_type,
    priority, rationale, parameters) into the same shape alert_feed.py's
    render_alert_card expects (an agent_actions row) — lets the preview
    reuse the exact same card styling as the logged/committed feed."""
    return {
        "vehicle_id": vehicle_id,
        "action_type": action["action_type"],
        "priority": action["priority"],
        "rationale": action["rationale"],
        "parameters": action.get("parameters") or {},
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ----------------------------------------------------------------------
# Live reasoning section
# ----------------------------------------------------------------------
def render_live_decision_section(conn, vehicle_id: str, profile_row: Dict[str, Any]) -> None:
    engine = get_decision_engine()
    session_key = f"agent_decision__{vehicle_id}"

    if engine is None:
        st.info(
            "Live agent reasoning is unavailable — set GEMINI_API_KEY to enable it. "
            "Showing this asset's decision history below."
        )
        return

    if st.button("Run agent reasoning", key=f"run_agent__{vehicle_id}"):
        with st.spinner(f"Reasoning over {vehicle_id}'s merged risk profile..."):
            st.session_state[session_key] = engine.decide_for_asset(profile_row)

    decision = st.session_state.get(session_key)
    if not decision:
        return

    st.markdown(f"**Assessment:** {decision.get('summary', '')}")
    for action in decision["actions"]:
        render_alert_card(decision_action_to_row(vehicle_id, action, status="pending — not yet logged"))

    col1, col2 = st.columns([1, 3])
    if col1.button("Execute & log these actions", key=f"execute_agent__{vehicle_id}", type="primary"):
        execute_decision(conn, decision)
        clear_all_caches()
        del st.session_state[session_key]
        st.success(f"Logged {len(decision['actions'])} action(s) for {vehicle_id}.")
        st.rerun()
    if col2.button("Discard", key=f"discard_agent__{vehicle_id}"):
        del st.session_state[session_key]
        st.rerun()


# ----------------------------------------------------------------------
# History section
# ----------------------------------------------------------------------
def render_agent_history_section(conn, vehicle_id: str) -> None:
    history_df = get_vehicle_actions(conn, vehicle_id)
    if history_df.empty:
        st.caption(f"No agent actions logged for {vehicle_id} yet.")
        return

    history_df = history_df.sort_values("created_at", ascending=False)
    for _, row in history_df.iterrows():
        render_alert_card(row.to_dict())


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_agent_recommendations(conn, vehicle_id: str, profile_row: Optional[Dict[str, Any]]) -> None:
    if profile_row is None:
        st.warning(f"No risk profile available for {vehicle_id} yet.")
        return

    st.subheader("Agent Reasoning")
    render_live_decision_section(conn, vehicle_id, profile_row)

    st.subheader("Decision History")
    render_agent_history_section(conn, vehicle_id)