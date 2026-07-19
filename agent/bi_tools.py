"""
agent/bi_tools.py

Read-only query tools exposed to the BI chat agent
(agent/bi_chat_engine.py). Every function here:

  - takes only JSON-serializable arguments (the model supplies these),
  - NEVER touches agent/actions.py's write path — this is a read-only
    surface, deliberately separate from anything that changes the
    fleet, matching the same detect-vs-decide separation the rest of
    this codebase already keeps (see models/risk_engine.py's docstring),
  - always returns a JSON-serializable dict, and never raises — bad
    input comes back as {"error": "..."} so the tool-calling loop can
    show the model its own mistake and retry, instead of the whole
    chat turn crashing.

Deliberately has ZERO dependency on dashboard/ or streamlit, even though
dashboard/utils.py already has similar helpers (fleet_summary_stats,
etc.) — agent/ has never depended on dashboard/ anywhere else in this
codebase (it's the other way around), and importing dashboard.utils
from here would be a reverse dependency for no real benefit. The small
amount of stats logic below is intentionally re-derived, not imported.

build_bi_tools(conn, profile_df, search_client=None) constructs the
actual Tool objects for one chat turn, closing over the already-computed
fleet profile (the dashboard already caches this via
dashboard/utils.get_fleet_profile, so tools reuse it instead of
re-fitting RUL for the whole fleet on every single tool call), the live
DB connection (needed for per-cycle telemetry, which isn't part of the
aggregate profile), and — optionally — a search_client for the one tool
that leaves the fleet database entirely.

Web search (`web_search` below) is the ONE tool in this registry that
touches the network. It's opt-in: build_bi_tools only registers it when
a search_client is passed in (agent/bi_chat_engine.py's
BIChatEngine.create(enable_web_search=True)), and even then the model is
instructed (BI_SYSTEM_PROMPT) to prefer every fleet-database tool first
and only reach for web_search when the question genuinely needs
information this system doesn't have — e.g. "what EV models should
replace EVR-0012" (see get_vehicle_metadata's docstring: make/model data
isn't wired in, so a real answer has to come from the web instead).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from agent.decision_engine import GeminiSearchClient, _attempt_json_repair, _strip_code_fence
from agent.prompts import build_web_search_prompt
from agent.tool_chat_engine import Tool

# Curated subset of risk_engine's full profile surfaced to the model —
# deliberately not every internal fitted parameter (fitted_a,
# fitted_decay_rate, r_squared, ...), matching the same "plain enough to
# reason over" spirit as dashboard/utils.py's STATUS_PLAIN_LABELS.
SUMMARY_COLUMNS = [
    "vehicle_id", "status", "overall_risk_level", "current_capacity_pct",
    "rul_cycles", "thermal_anomaly_count", "critical_temp_count",
    "max_security_severity", "unticketed_command_count", "suspicious_command_count",
    "fast_charge_frequency_pct", "mean_dod_pct", "charge_stress_score",
    "stress_trend", "suggested_policy",
]

TIMESERIES_METRICS = ("capacity_pct_of_rated", "temperature_c", "soc_pct", "dod_pct", "voltage")

RANKABLE_METRICS = (
    "rul_cycles", "current_capacity_pct", "charge_stress_score",
    "thermal_anomaly_count", "mean_dod_pct", "fast_charge_frequency_pct",
    "unticketed_command_count",
)


def _row_to_summary(row: "pd.Series") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in SUMMARY_COLUMNS:
        if col not in row:
            continue
        val = row[col]
        out[col] = None if (isinstance(val, float) and pd.isnull(val)) else val
    return out


def _infer_params(fn: Callable[..., Any]) -> Dict[str, str]:
    """inspect.signature() returns *string* annotations here (not type
    objects) because this module has `from __future__ import annotations`
    (PEP 563) -- so `p.annotation` is already e.g. "str", not the `str`
    class, and calling .__name__ on it would fail. Handle both forms
    defensively in case this is ever called against a function from a
    module without that import."""
    sig = inspect.signature(fn)
    params: Dict[str, str] = {}
    for name, p in sig.parameters.items():
        if p.annotation is inspect._empty:
            type_name = "any"
        elif isinstance(p.annotation, str):
            type_name = p.annotation
        else:
            type_name = getattr(p.annotation, "__name__", str(p.annotation))
        default = "" if p.default is inspect._empty else f", default={p.default!r}"
        params[name] = f"{type_name}{default}"
    return params


def build_bi_tools(
    conn,
    profile_df: pd.DataFrame,
    search_client: Optional[GeminiSearchClient] = None,
) -> Dict[str, Tool]:
    def get_fleet_summary() -> Dict[str, Any]:
        """Fleet-wide summary: total vehicle count, risk-level distribution
        (minimal/low/medium/high counts), how many need maintenance, how
        many have an active security signal, and the mean charge stress
        score across the fleet."""
        if profile_df.empty:
            return {"total_vehicles": 0}
        risk_counts = profile_df["overall_risk_level"].value_counts().to_dict()
        mean_stress = profile_df["charge_stress_score"].mean() if "charge_stress_score" in profile_df else None
        return {
            "total_vehicles": int(len(profile_df)),
            "risk_level_counts": {str(k): int(v) for k, v in risk_counts.items()},
            "vehicles_needing_maintenance": int(profile_df["status"].isin(["degraded", "critical"]).sum()),
            "vehicles_with_active_security_signal": int((profile_df["max_security_severity"] != "none").sum()),
            "mean_charge_stress_score": (
                round(float(mean_stress), 1) if mean_stress is not None and pd.notnull(mean_stress) else None
            ),
        }

    def list_vehicles(
        status: str = "", risk_level: str = "", security_severity: str = "", limit: int = 50,
    ) -> Dict[str, Any]:
        """Lists vehicles matching optional filters: status (healthy/watch/
        degraded/critical), risk_level (minimal/low/medium/high), and/or
        security_severity (none/low/medium/high). Leave a filter as an
        empty string to not filter on it."""
        df = profile_df
        if df.empty:
            return {"vehicles": [], "matched": 0}
        if status:
            df = df[df["status"] == status]
        if risk_level:
            df = df[df["overall_risk_level"] == risk_level]
        if security_severity:
            df = df[df["max_security_severity"] == security_severity]
        rows = [_row_to_summary(r) for _, r in df.head(max(1, min(limit, 200))).iterrows()]
        return {"vehicles": rows, "matched": int(len(df))}

    def get_vehicle_profile(vehicle_id: str) -> Dict[str, Any]:
        """Full risk-profile summary for one vehicle by ID."""
        match = profile_df[profile_df["vehicle_id"] == vehicle_id]
        if match.empty:
            return {"error": f"no vehicle '{vehicle_id}' in the current fleet profile"}
        return _row_to_summary(match.iloc[0])

    def compare_vehicles(vehicle_ids: str) -> Dict[str, Any]:
        """Side-by-side profile summaries for several vehicles at once.
        vehicle_ids: comma-separated IDs, e.g. 'EVR-0001,EVR-0007'."""
        ids = [v.strip() for v in vehicle_ids.split(",") if v.strip()]
        if not ids:
            return {"error": "no vehicle_ids provided"}
        rows, missing = [], []
        for vid in ids:
            match = profile_df[profile_df["vehicle_id"] == vid]
            (missing if match.empty else rows).append(vid if match.empty else _row_to_summary(match.iloc[0]))
        result: Dict[str, Any] = {"vehicles": rows}
        if missing:
            result["not_found"] = missing
        return result

    def rank_vehicles(metric: str, ascending: bool = True, limit: int = 10) -> Dict[str, Any]:
        """Ranks vehicles by one numeric profile metric. metric must be one
        of: rul_cycles, current_capacity_pct, charge_stress_score,
        thermal_anomaly_count, mean_dod_pct, fast_charge_frequency_pct,
        unticketed_command_count. ascending=True means lowest-first (useful
        for e.g. lowest rul_cycles = most in need of attention)."""
        if metric not in RANKABLE_METRICS:
            return {"error": f"unknown metric '{metric}'. Must be one of: {', '.join(RANKABLE_METRICS)}"}
        if profile_df.empty or metric not in profile_df.columns:
            return {"vehicles": []}
        df = profile_df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        rows = [_row_to_summary(r) for _, r in df.head(max(1, min(limit, 50))).iterrows()]
        return {"vehicles": rows, "metric": metric, "ascending": ascending}

    def get_vehicle_timeseries(vehicle_id: str, metric: str) -> Dict[str, Any]:
        """Per-cycle history of one telemetry metric for one vehicle. metric
        must be one of: capacity_pct_of_rated, temperature_c, soc_pct,
        dod_pct, voltage."""
        if metric not in TIMESERIES_METRICS:
            return {"error": f"unknown metric '{metric}'. Must be one of: {', '.join(TIMESERIES_METRICS)}"}
        from ingestion.db import get_telemetry_for_vehicle

        rows = get_telemetry_for_vehicle(conn, vehicle_id)
        if not rows:
            return {"error": f"no telemetry found for '{vehicle_id}'"}
        return {
            "vehicle_id": vehicle_id, "metric": metric,
            "points": [{"cycle": r["cycle"], "value": r[metric]} for r in rows],
        }

    def compare_vehicle_timeseries(vehicle_ids: str, metric: str) -> Dict[str, Any]:
        """Per-cycle history of one telemetry metric across several vehicles
        at once — use this (not repeated get_vehicle_timeseries calls) when
        the fleet manager wants a multi-vehicle trend comparison.
        vehicle_ids: comma-separated, e.g. 'EVR-0001,EVR-0007'. metric: one
        of capacity_pct_of_rated, temperature_c, soc_pct, dod_pct, voltage."""
        if metric not in TIMESERIES_METRICS:
            return {"error": f"unknown metric '{metric}'. Must be one of: {', '.join(TIMESERIES_METRICS)}"}
        from ingestion.db import get_telemetry_for_vehicle

        ids = [v.strip() for v in vehicle_ids.split(",") if v.strip()]
        series: Dict[str, List[Dict[str, Any]]] = {}
        missing = []
        for vid in ids:
            rows = get_telemetry_for_vehicle(conn, vid)
            if not rows:
                missing.append(vid)
                continue
            series[vid] = [{"cycle": r["cycle"], "value": r[metric]} for r in rows]
        result: Dict[str, Any] = {"metric": metric, "series": series}
        if missing:
            result["not_found"] = missing
        return result

    def get_vehicle_metadata(vehicle_id: str) -> Dict[str, Any]:
        """Make/model/purchase-date metadata for a vehicle. NOT YET
        AVAILABLE in this deployment — call this to check before assuming
        it exists; it returns an honest 'unavailable' marker rather than a
        fabricated make/model. For "what should replace this vehicle"
        style questions, base the assessment on RUL/thermal/charge-stress
        signals from the fleet tools, and use web_search (if available)
        for actual replacement model suggestions."""
        return {
            "available": False,
            "message": (
                "Vehicle make/model/purchase-date metadata isn't wired into "
                "VoltSentinel yet — only battery telemetry and risk signals "
                "are available. Base replacement suggestions on RUL, thermal, "
                "and charge-stress signals, and use web_search for actual "
                "replacement vehicle/model recommendations if that tool is available."
            ),
        }

    def web_search(query: str) -> Dict[str, Any]:
        """Searches the public web for information the fleet database
        cannot provide — e.g. replacement EV or battery-pack models,
        industry specifications, or general battery/charging best
        practices. ONLY use this when the fleet-data tools above genuinely
        can't answer the question; never use it for anything already
        answerable from get_fleet_summary / list_vehicles / get_vehicle_profile
        / rank_vehicles / get_vehicle_timeseries, and never for questions
        unrelated to this fleet's operations. Returns {"answer": str,
        "sources": [str, ...]} on success, or {"error": str} if web search
        isn't enabled for this session or the search itself failed."""
        if search_client is None:
            return {"error": "web search is not enabled for this session"}

        prompt = build_web_search_prompt(query)
        try:
            raw = search_client.generate(prompt)
        except Exception as e:
            return {"error": f"web search request failed: {e}"}

        text = _strip_code_fence(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            repaired = _attempt_json_repair(text)
            if repaired is None:
                return {"error": "web search returned an unparseable response"}
            data = json.loads(repaired)

        if not isinstance(data, dict):
            return {"error": "web search returned a malformed response"}

        answer = data.get("answer")
        if not isinstance(answer, str) or not answer:
            return {"error": "web search returned no answer"}

        sources = data.get("sources")
        sources = [str(s) for s in sources] if isinstance(sources, list) else []

        return {"answer": answer, "sources": sources}

    fns = [
        get_fleet_summary, list_vehicles, get_vehicle_profile, compare_vehicles,
        rank_vehicles, get_vehicle_timeseries, compare_vehicle_timeseries,
        get_vehicle_metadata,
    ]
    # web_search is only ever registered — i.e. only ever visible to the
    # model as an available tool at all — when a search_client was passed
    # in. This is the actual on/off switch: an unregistered tool can't be
    # called regardless of what the system prompt says, so disabling web
    # search for a session/user doesn't rely on prompt compliance alone.
    if search_client is not None:
        fns.append(web_search)

    tools: Dict[str, Tool] = {}
    for fn in fns:
        tools[fn.__name__] = Tool(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0] or fn.__name__,
            parameters=_infer_params(fn),
            fn=fn,
        )
    return tools