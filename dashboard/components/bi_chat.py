"""
dashboard/components/bi_chat.py

The "ask the fleet in plain English" tab: a few always-on default
charts (fleet risk mix, RUL distribution, charge-stress vs thermal
scatter) plus a chat box wired to agent/bi_chat_engine.py's read-only
tool-calling loop. The fleet manager can ask for comparisons, rankings,
or trends in plain English and get back a short answer plus, when the
question is comparative/trend-based, a chart built from whatever data
the agent's tool calls actually returned — the chart type and fields
come from the model's own answer (agent/bi_chat_engine.py's chart
spec), not a fixed chart per query.

Same split as every other dashboard/components/*.py file:
    - build_*_chart(...)      -- pure, alt.Chart-returning, no st.*.
    - build_chart_from_spec() -- same, but for the agent's dynamic spec
      instead of a fixed dataset.
    - render_bi_chat(...)     -- the only function here touching st.*.

Read-only by design: this tab has no access to agent/actions.py's write
path (see agent/bi_chat_engine.py's docstring) — it can tell the fleet
manager what it found, never change anything in the fleet.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.utils import RISK_LEVEL_COLORS, RISK_LEVEL_ORDER, risk_level_label

CHAT_HISTORY_KEY = "bi_chat_history"


# ----------------------------------------------------------------------
# Cached engine -- one Gemini client per session, mirrors
# agent_recommendations.py's get_decision_engine() pattern exactly.
# ----------------------------------------------------------------------
@st.cache_resource
def get_bi_chat_engine():
    from agent.bi_chat_engine import BIChatEngine

    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        return BIChatEngine.create()
    except Exception:
        return None


# ----------------------------------------------------------------------
# Default charts -- pure builders, no st.*
# ----------------------------------------------------------------------
def build_risk_mix_chart(profile_df: pd.DataFrame) -> alt.Chart:
    counts = profile_df["overall_risk_level"].value_counts()
    data = pd.DataFrame({
        "risk_level": [risk_level_label(l) for l in RISK_LEVEL_ORDER],
        "count": [int(counts.get(l, 0)) for l in RISK_LEVEL_ORDER],
        "color": [RISK_LEVEL_COLORS[l] for l in RISK_LEVEL_ORDER],
    })
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("risk_level:N", title="Risk level", sort=[risk_level_label(l) for l in RISK_LEVEL_ORDER]),
            y=alt.Y("count:Q", title="Vehicles"),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["risk_level", "count"],
        )
        .properties(height=260, title="Fleet risk mix")
    )


def build_rul_distribution_chart(profile_df: pd.DataFrame) -> alt.Chart:
    data = profile_df.dropna(subset=["rul_cycles"])
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("rul_cycles:Q", bin=alt.Bin(maxbins=20), title="Projected RUL (cycles)"),
            y=alt.Y("count()", title="Vehicles"),
            tooltip=["count()"],
        )
        .properties(height=260, title="Remaining useful life distribution")
    )


def build_stress_vs_thermal_chart(profile_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(profile_df)
        .mark_circle(size=90)
        .encode(
            x=alt.X("charge_stress_score:Q", title="Charge stress score"),
            y=alt.Y("thermal_anomaly_count:Q", title="Thermal anomalies"),
            color=alt.Color(
                "overall_risk_level:N", title="Risk",
                scale=alt.Scale(domain=list(RISK_LEVEL_ORDER),
                                 range=[RISK_LEVEL_COLORS[l] for l in RISK_LEVEL_ORDER]),
            ),
            tooltip=["vehicle_id", "charge_stress_score", "thermal_anomaly_count", "overall_risk_level"],
        )
        .properties(height=280, title="Charging stress vs. thermal anomalies")
        .interactive()
    )


def render_default_charts(profile_df: pd.DataFrame) -> None:
    if profile_df.empty:
        st.info("No fleet data yet — default charts will appear once vehicles are ingested.")
        return
    col1, col2 = st.columns(2)
    col1.altair_chart(build_risk_mix_chart(profile_df), use_container_width=True)
    col2.altair_chart(build_rul_distribution_chart(profile_df), use_container_width=True)
    st.altair_chart(build_stress_vs_thermal_chart(profile_df), use_container_width=True)


# ----------------------------------------------------------------------
# Dynamic chart -- built from whatever spec the agent's final_answer
# returned (agent/bi_chat_engine.py's CHART_TYPES: bar/line/scatter/table)
# ----------------------------------------------------------------------
def build_chart_from_spec(spec: Dict[str, Any]) -> Optional[alt.Chart]:
    """Returns None for a "table" spec (rendered as a dataframe by the
    caller instead), for a spec with no usable data, or for any field
    name the model invented that isn't actually in its own data — never
    raises, since a slightly malformed spec should just fall back to
    plain text, not break the chat turn."""
    data = pd.DataFrame(spec.get("data") or [])
    if data.empty:
        return None

    chart_type = spec.get("type")
    x_field = spec.get("x_field")
    y_field = spec.get("y_field")
    series_field = spec.get("series_field")
    title = spec.get("title") or ""

    if chart_type not in ("bar", "line", "scatter"):
        return None
    if x_field not in data.columns or y_field not in data.columns:
        return None

    x_type = "Q" if chart_type == "line" else "N"
    encodings: Dict[str, Any] = {
        "x": alt.X(f"{x_field}:{x_type}", title=x_field),
        "y": alt.Y(f"{y_field}:Q", title=y_field),
        "tooltip": list(data.columns),
    }
    if series_field and series_field in data.columns:
        encodings["color"] = alt.Color(f"{series_field}:N", title=series_field)

    if chart_type == "bar":
        mark = alt.Chart(data).mark_bar()
    elif chart_type == "line":
        mark = alt.Chart(data).mark_line(point=True)
    else:
        mark = alt.Chart(data).mark_circle(size=90)

    return mark.encode(**encodings).properties(height=300, title=title).interactive()


def render_agent_message(message: Dict[str, Any]) -> None:
    st.markdown(message.get("text", ""))
    chart_spec = message.get("chart")
    if not chart_spec:
        return
    if chart_spec.get("type") == "table":
        st.dataframe(pd.DataFrame(chart_spec.get("data") or []), use_container_width=True, hide_index=True)
        return
    chart = build_chart_from_spec(chart_spec)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------------------------
# Top-level entrypoint -- the only function app.py needs to call
# ----------------------------------------------------------------------
def render_bi_chat(conn, profile_df: pd.DataFrame) -> None:
    st.subheader("Fleet BI")
    st.caption(
        "Default charts below, or ask a question — e.g. \"compare EVR-0001 and EVR-0007's "
        "charging stress\" or \"which vehicles are lowest on RUL?\""
    )

    render_default_charts(profile_df)
    st.divider()

    engine = get_bi_chat_engine()
    history: List[Dict[str, Any]] = st.session_state.setdefault(CHAT_HISTORY_KEY, [])

    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_agent_message(message)
            else:
                st.markdown(message["text"])

    if engine is None:
        st.info("Set GEMINI_API_KEY to enable the BI chat — default charts above still work without it.")
        return

    question = st.chat_input("Ask about the fleet...")
    if not question:
        return

    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the fleet data..."):
            chat_context = [{"role": m["role"], "text": m["text"]} for m in history[:-1]]
            answer = engine.ask(conn, profile_df, question, chat_history=chat_context)
        render_agent_message(answer)

    history.append({"role": "assistant", "text": answer["text"], "chart": answer.get("chart")})