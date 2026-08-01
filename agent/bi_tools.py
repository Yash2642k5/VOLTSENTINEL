from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from agent.decision_engine import GeminiSearchClient, _attempt_json_repair, _strip_code_fence
from agent.prompts import build_web_search_prompt
from agent.tool_chat_engine import Tool

SUMMARY_COLUMNS = [
    "vehicle_id", "status", "overall_risk_level", "current_capacity_pct",
    "rul_cycles", "thermal_anomaly_count", "critical_temp_count",
    "max_security_severity", "unticketed_command_count", "suspicious_command_count",
    "fast_charge_frequency_pct", "mean_dod_pct", "charge_stress_score",
    "stress_trend", "suggested_policy",
    "missing_cycle_count", "is_stale", "hours_since_last_reading", "out_of_range_jump_count",
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
        match = profile_df[profile_df["vehicle_id"] == vehicle_id]
        if match.empty:
            return {"error": f"no vehicle '{vehicle_id}' in the current fleet profile"}
        return _row_to_summary(match.iloc[0])

    def compare_vehicles(vehicle_ids: str) -> Dict[str, Any]:
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
        if metric not in RANKABLE_METRICS:
            return {"error": f"unknown metric '{metric}'. Must be one of: {', '.join(RANKABLE_METRICS)}"}
        if profile_df.empty or metric not in profile_df.columns:
            return {"vehicles": []}
        df = profile_df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        rows = [_row_to_summary(r) for _, r in df.head(max(1, min(limit, 50))).iterrows()]
        return {"vehicles": rows, "metric": metric, "ascending": ascending}

    def get_vehicle_timeseries(vehicle_id: str, metric: str) -> Dict[str, Any]:
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
        from ingestion.db import get_vehicle_metadata as _get_vehicle_metadata

        row = _get_vehicle_metadata(conn, vehicle_id)
        if row is None:
            return {
                "available": False,
                "message": f"No asset-registry entry for '{vehicle_id}'.",
            }
        return {
            "available": True,
            "vehicle_id": row["vehicle_id"],
            "make": row["make"],
            "model": row["model"],
            "vin": row["vin"],
            "purchase_date": row["purchase_date"],
            "warranty_expiry_date": row["warranty_expiry_date"],
        }

    def get_reliability_metrics(vehicle_id: str) -> Dict[str, Any]:
        from ingestion.db import get_all_vehicle_ids
        from models.reliability_metrics import ReliabilityAnalyzer

        if vehicle_id not in get_all_vehicle_ids(conn):
            return {"error": f"no vehicle '{vehicle_id}' in the fleet"}
        return ReliabilityAnalyzer().analyze_vehicle(conn, vehicle_id).to_dict()

    def web_search(query: str) -> Dict[str, Any]:
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
        get_vehicle_metadata, get_reliability_metrics,
    ]

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