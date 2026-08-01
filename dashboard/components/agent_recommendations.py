from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from agent.actions import execute_decision, release_vehicle_quarantine
from agent.decision_engine import DecisionEngine
from dashboard.components.alert_feed import render_alert_card
from dashboard.utils import clear_all_caches, get_vehicle_actions

@st.cache_resource
def get_decision_engine() -> Optional[DecisionEngine]:
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        return DecisionEngine.create()
    except Exception:
        return None

#helpers
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

def render_quarantine_controls(conn, vehicle_id: str) -> None:
    """No-ops entirely if the vehicle isn't currently quarantined — this
    banner should only ever appear when agent/actions.py's
    quarantine_vehicle has actually fired for this asset."""
    from ingestion.db import get_quarantine_status

    status = get_quarantine_status(conn, vehicle_id)
    if not status or not status["active"]:
        return

    st.error(
        f"**{vehicle_id} is quarantined.** Any further unticketed BMS command for this "
        f"vehicle is being rejected outright at ingestion. Reason: {status['reason']}",
        icon="🔒",
    )

    col1, col2 = st.columns([3, 1])
    released_by = col1.text_input(
        "Your name (recorded in the audit log as who released this)",
        key=f"release_by__{vehicle_id}",
    )
    if col2.button(
        "Release quarantine", key=f"release_quarantine__{vehicle_id}",
        type="primary", disabled=not released_by.strip(),
    ):
        release_vehicle_quarantine(conn, vehicle_id, released_by=released_by.strip())
        clear_all_caches()
        st.success(f"Quarantine released for {vehicle_id} by {released_by.strip()}.")
        st.rerun()

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
        if action["action_type"] == "quarantine_vehicle":
            st.caption(
                "⚠️ This action will lock out further unticketed BMS commands for this "
                "vehicle immediately upon execution — it does not affect a moving vehicle "
                "directly, but it does change what the vehicle will accept going forward."
            )

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

def render_agent_history_section(conn, vehicle_id: str) -> None:
    history_df = get_vehicle_actions(conn, vehicle_id)
    if history_df.empty:
        st.caption(f"No agent actions logged for {vehicle_id} yet.")
        return

    history_df = history_df.sort_values("created_at", ascending=False)
    for _, row in history_df.iterrows():
        render_alert_card(row.to_dict())

def render_agent_recommendations(conn, vehicle_id: str, profile_row: Optional[Dict[str, Any]]) -> None:
    if profile_row is None:
        st.warning(f"No risk profile available for {vehicle_id} yet.")
        return

    render_quarantine_controls(conn, vehicle_id)

    st.subheader("Agent Reasoning")
    render_live_decision_section(conn, vehicle_id, profile_row)

    st.subheader("Decision History")
    render_agent_history_section(conn, vehicle_id)