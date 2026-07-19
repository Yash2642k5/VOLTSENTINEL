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

Web search: BIChatEngine can optionally reach the open web through
agent/bi_tools.py's `web_search` tool, backed by a SEPARATE
GeminiSearchClient (agent/decision_engine.py) — a different client
object than `client` above, built with its own narrow system prompt
(BI_WEB_SEARCH_SYSTEM_PROMPT, agent/prompts.py) rather than
RESEARCH_SYSTEM_PROMPT (which is scoped to the Agent tab's per-asset
research flow and expects a different response schema). This is
opt-in: enable_web_search=False by default, matching
DecisionEngine.create(enable_research=...)'s pattern. When disabled,
`web_search` is never even registered in the tool catalogue
build_bi_tools returns (see bi_tools.py), so the model can't call it
regardless of what the prompt says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.bi_tools import build_bi_tools
from agent.decision_engine import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, GeminiClient, GeminiSearchClient
from agent.prompts import BI_WEB_SEARCH_SYSTEM_PROMPT
from agent.tool_chat_engine import run_tool_loop

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
- Prefer the fleet-data tools (get_fleet_summary, list_vehicles, get_vehicle_profile, \
compare_vehicles, rank_vehicles, get_vehicle_timeseries, compare_vehicle_timeseries) for \
anything answerable from this fleet's own data — they're always faster and more \
reliable than a search.
- If (and only if) a "web_search" tool is available in your tool list AND the question \
genuinely needs information this fleet database cannot provide — e.g. "what EV models \
should replace EVR-0012", industry specifications, or general best practices — call \
web_search. If web_search is not in your tool list, or the question doesn't need it, do \
not attempt to search the web; answer from fleet data alone, or explain plainly that the \
information isn't available in this deployment.
- If a question has no connection to fleet \
operations at all (general trivia, coding help, unrelated topics), do not call a \
tool — respond with a final_answer whose "text" is: "I can only help with questions \
about your fleet's battery health, charging, security, and operations." and \
"chart": null.
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
signals, then use web_search (if available) for actual replacement model suggestions.
- If you use web_search, cite what you found in "text" and don't present it as fleet \
data — make clear it's external/general information, not something from this fleet's \
own telemetry.
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
    search_client: Optional[GeminiSearchClient] = None   # None => web_search tool disabled
    max_tool_turns: int = 6
    retry_backoff_seconds: float = 1.0

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        enable_web_search: bool = False,
    ) -> "BIChatEngine":
        """enable_web_search=False by default — matches
        DecisionEngine.create(enable_research=...)'s opt-in pattern. Even
        when True, this only ever populates search_client; the actual
        on/off switch is whether build_bi_tools() registers the
        `web_search` tool at all (see bi_tools.py), not prompt wording."""
        client = GeminiClient(
            api_key=api_key, model_name=model_name, temperature=temperature,
            system_instruction=BI_SYSTEM_PROMPT,
        )
        search_client = (
            GeminiSearchClient(
                api_key=api_key, model_name=model_name, temperature=temperature,
                system_instruction=BI_WEB_SEARCH_SYSTEM_PROMPT,
            )
            if enable_web_search else None
        )
        return cls(client=client, search_client=search_client)

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
        tools = build_bi_tools(conn, profile_df, search_client=self.search_client)

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