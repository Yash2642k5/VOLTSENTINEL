"""
dashboard/components/alert_feed.py

Fleet-wide "live alert feed" (project doc §5.1's Presentation row):
a reverse-chronological ticker of everything agent/decision_engine.py
has logged via agent/actions.py. Distinct from agent_recommendations.py,
which shows one asset's full reasoning; this shows the fleet-wide stream
of what the agent has actually done, most recent first.

Same split as the other components:
    - build_alert_feed_view(...) is a pure transform (filter/sort/slice),
    testable without a running Streamlit app.
    - render_alert_card(...) / render_alert_feed(...) are the only
    functions here that call st.*.

Excludes action_type == "no_action" by default — those are logged for
audit completeness (agent/actions.py's docstring: "a decision is still
a decision"), but a live demo feed showing "EVR-0004: nothing to do"
repeatedly would bury the events that actually matter. exclude_no_action
is a parameter, not a hardcoded choice, so a "show everything" toggle
is one line away in app.py if wanted later.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.utils import (
    PRIORITY_COLORS,
    action_type_label,
    format_relative_time,
    get_recent_actions,
)

PRIORITY_ORDER = ("critical", "high", "medium", "low")

ACTION_TYPE_ICONS: Dict[str, str] = {
    "maintenance_trigger": "🔧",
    "charge_policy_recommendation": "⚡",
    "security_escalation": "🚨",
    "fleet_manager_notification": "📣",
    "no_action": "✅",
}

DEFAULT_MAX_ITEMS = 25


# ----------------------------------------------------------------------
# Pure data prep
# ----------------------------------------------------------------------
def build_alert_feed_view(
    actions_df: pd.DataFrame,
    max_items: int = DEFAULT_MAX_ITEMS,
    priority_filter: Optional[List[str]] = None,
    action_type_filter: Optional[List[str]] = None,
    vehicle_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Filters, sorts newest-first, and caps the feed. All filters are
    optional — passing None means 'no restriction on this dimension',
    not 'match nothing', so callers can apply only the filters they need."""
    if actions_df.empty:
        return actions_df

    df = actions_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at", ascending=False)

    if priority_filter is not None:
        df = df[df["priority"].isin(priority_filter)]
    if action_type_filter is not None:
        df = df[df["action_type"].isin(action_type_filter)]
    if vehicle_filter is not None:
        df = df[df["vehicle_id"] == vehicle_filter]

    return df.head(max_items)


def priority_counts(actions_df: pd.DataFrame) -> Dict[str, int]:
    if actions_df.empty:
        return {p: 0 for p in PRIORITY_ORDER}
    counts = actions_df["priority"].value_counts()
    return {p: int(counts.get(p, 0)) for p in PRIORITY_ORDER}


def _parse_parameters(raw: Any) -> Dict[str, Any]:
    """agent/actions.py stores parameters as a JSON string (json.dumps
    before the sqlite insert) — this undoes that for display."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_priority_summary(actions_df: pd.DataFrame) -> None:
    counts = priority_counts(actions_df)
    badges = " ".join(
        f"<span style='background-color:{PRIORITY_COLORS[p]};color:white;"
        f"padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;"
        f"margin-right:6px;'>{p.title()}: {counts[p]}</span>"
        for p in PRIORITY_ORDER
    )
    st.markdown(badges, unsafe_allow_html=True)


def render_alert_card(row: Dict[str, Any]) -> None:
    priority = row.get("priority", "low")
    color = PRIORITY_COLORS.get(priority, "#757575")
    icon = ACTION_TYPE_ICONS.get(row.get("action_type"), "•")
    label = action_type_label(row.get("action_type"))
    when = format_relative_time(row.get("created_at"))

    st.markdown(
        f"""<div style='border-left:4px solid {color};padding:8px 14px;margin-bottom:8px;
        background-color:rgba(128,128,128,0.06);border-radius:4px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-weight:600;'>{icon} {label} — {row.get('vehicle_id', 'UNKNOWN')}</span>
            <span style='color:{color};font-weight:600;font-size:0.8rem;'>{priority.upper()}</span>
        </div>
        <div style='font-size:0.9rem;margin-top:4px;'>{row.get('rationale', '')}</div>
        <div style='font-size:0.75rem;color:#9E9E9E;margin-top:4px;'>{when} · {row.get('status', '')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    parameters = _parse_parameters(row.get("parameters"))
    if parameters:
        with st.expander("Details", expanded=False):
            st.json(parameters)


# ----------------------------------------------------------------------
# Top-level entrypoint — the only function app.py needs to call
# ----------------------------------------------------------------------
def render_alert_feed(
    conn,
    max_items: int = DEFAULT_MAX_ITEMS,
    exclude_no_action: bool = True,
    show_filters: bool = True,
) -> None:
    actions_df = get_recent_actions(conn, exclude_no_action=exclude_no_action)

    if actions_df.empty:
        st.info("No agent actions logged yet.")
        return

    render_priority_summary(actions_df)

    priority_filter = None
    action_type_filter = None
    if show_filters:
        available_priorities = [p for p in PRIORITY_ORDER if p in actions_df["priority"].unique()]
        available_types = sorted(actions_df["action_type"].unique())

        cols = st.columns(2)
        selected_priorities = cols[0].multiselect(
            "Priority", available_priorities, default=available_priorities, key="alert_feed_priority_filter"
        )
        selected_types = cols[1].multiselect(
            "Action type", available_types,
            default=available_types,
            format_func=action_type_label,
            key="alert_feed_type_filter",
        )
        priority_filter = selected_priorities or None
        action_type_filter = selected_types or None

    view = build_alert_feed_view(
        actions_df, max_items=max_items,
        priority_filter=priority_filter, action_type_filter=action_type_filter,
    )

    if view.empty:
        st.info("No alerts match the current filters.")
        return

    for _, row in view.iterrows():
        render_alert_card(row.to_dict())