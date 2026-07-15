"""
agent/bi_chat_engine.py

The "ask the fleet in plain English" surface. Wraps the generic
agent/tool_chat_engine.py loop with a read-only tool registry
(agent/bi_tools.py) and a system prompt that defines the expected final
answer shape (short text + an optional chart spec) — so
dashboard/components/bi_chat.py can render whatever chart the model
decides actually fits the question, instead of only ever showing fixed
default charts.

Deliberately read-only: this engine has no access to agent/actions.py's
write path at all. The planned incident/security chat reuses the exact
same tool_chat_engine.py loop, but with a tool registry that also
exposes actions as *proposals* the fleet manager must explicitly accept
before anything executes — kept as a clearly separate engine/tool
registry rather than blurred into this one, so "ask a question about
the fleet" can never accidentally trigger a write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .bi_tools import build_bi_tools
from .decision_engine import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, GeminiClient
from .tool_chat_engine import run_tool_loop

CHART_TYPES = ("bar", "line", "scatter", "table")

BI_SYSTEM_PROMPT = """You are VoltSentinel's fleet BI assistant. A fleet manager asks \
you plain-English questions about their EV battery fleet — comparing vehicles, \
finding which ones need attention, spotting trends — and you answer using the tools \
available to you. You have no direct database access; everything you know about the \
fleet comes from calling the tools you're given.

You work in a strict turn-by-turn protocol. On EVERY turn, respond with EXACTLY one \
JSON object and nothing else — no prose before or after, no markdown fences.

To call a tool:
{"tool_call": {"name": "<tool name>", "arguments": {"<param>": <value>, ...}}}

To give your final answer once you have enough information:
{"final_answer": {
  "text": "<a short, plain-English answer -- 1-4 sentences, grounded in real numbers>",
  "chart": null
    OR
  {
    "type": "<bar | line | scatter | table>",
    "title": "<short chart title>",
    "x_field": "<field name in data to use as the x-axis / row key>",
    "y_field": "<field name in data to plot as the value>",
    "series_field": "<optional field name to color/split by, e.g. vehicle_id, or null>",
    "data": [ {"<x_field>": ..., "<y_field>": ..., "<series_field if used>": ...}, ... ]
  }
}}

Guidelines:
- Only include a chart for genuinely comparative or trend-based questions ("compare", \
"over time", "which vehicles", "show me", "rank"). A single fact lookup doesn't need \
one — set "chart": null.
- Build "data" yourself by reshaping whatever the tools returned into the flat \
{x_field, y_field, series_field} record shape the chart needs. Every value in "data" \
must trace back to an actual tool result.
- Never invent a vehicle_id, metric value, or fleet statistic that didn't come from a \
tool result. If you don't have enough information yet, call another tool instead of \
guessing.
- If a tool result contains an "error" key, read it, adjust your arguments, and try \
again rather than repeating the same failing call.
- Vehicle make/model/purchase-date metadata is not currently available — call \
get_vehicle_metadata to confirm this if asked, and for "which vehicles should I \
replace" style questions, answer using RUL, thermal-anomaly, and charge-stress \
signals instead, saying plainly that make/model data isn't wired in yet.
- Keep "text" specific: cite the actual numbers you found, not generic statements."""


def _validate_final_answer(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive normalization of the model's final_answer payload — never
    raises, since a malformed chart spec should just degrade to no chart
    rather than break the chat turn."""
    text = raw.get("text")
    if not text or not isinstance(text, str):
        text = "(no answer text provided)"

    chart = raw.get("chart")
    if chart is not None:
        if not isinstance(chart, dict):
            chart = None
        else:
            chart_type = chart.get("type")
            data = chart.get("data")
            if chart_type not in CHART_TYPES or not isinstance(data, list) or not data:
                chart = None

    return {"text": text, "chart": chart}


@dataclass
class BIChatEngine:
    client: GeminiClient
    max_tool_turns: int = 6
    retry_backoff_seconds: float = 1.0

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> "BIChatEngine":
        client = GeminiClient(
            api_key=api_key, model_name=model_name, temperature=temperature,
            system_instruction=BI_SYSTEM_PROMPT,
        )
        return cls(client=client)

    def ask(
        self,
        conn,
        profile_df: pd.DataFrame,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """chat_history: prior turns in this conversation as
        [{"role": "user"|"assistant", "text": ...}, ...] — gives the model
        conversational context ("what about EVR-0002 too?"). Each call still
        runs its own fresh tool-call loop against the CURRENT conn/
        profile_df, so answers reflect live data even mid-conversation
        rather than data snapshotted at conversation start.

        Always returns {"text": str, "chart": dict | None} — never raises."""
        tools = build_bi_tools(conn, profile_df)

        transcript: List[str] = []
        for turn in (chat_history or []):
            speaker = "Fleet manager" if turn["role"] == "user" else "You"
            transcript.append(f"{speaker} (earlier in this conversation): {turn['text']}")
        transcript.append(f"Fleet manager asks: {question}")

        result = run_tool_loop(
            client=self.client,
            tools=tools,
            transcript=transcript,
            max_tool_turns=self.max_tool_turns,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        return _validate_final_answer(result)