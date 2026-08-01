from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .actions import execute_decision
from .prompts import (
    ACTION_TYPES,
    PRIORITY_LEVELS,
    RESEARCH_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_decision_prompt,
    build_research_prompt,
)

DEFAULT_MODEL_NAME = "gemini-3.6-flash"
DEFAULT_MODEL_NAME_2 = "gemini-3.5-flash"
DEFAULT_TEMPERATURE = 0.2
MAX_ATTEMPTS = 2



class DecisionParseError(Exception):
    """Raised internally when the LLM response can't be turned into a
    valid decision object. Always caught inside decide_for_asset — never
    escapes to the caller, since a single asset's failure shouldn't kill
    a fleet run."""


class QuotaExceededError(Exception):

    def __init__(self, message: str, retry_delay_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds

_NO_SDK_RETRY_REQUEST_OPTIONS = {"retry": None}


def _extract_retry_delay_seconds(error: Exception) -> Optional[float]:
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(error))
    if match:
        return float(match.group(1))
    match = re.search(r"seconds:\s*(\d+)", str(error))
    if match:
        return float(match.group(1))
    return None


def _is_quota_error(error: Exception) -> bool:
    if type(error).__name__ in ("ResourceExhausted", "TooManyRequests"):
        return True
    text = str(error)
    return "429" in text or "quota" in text.lower() or "RESOURCE_EXHAUSTED" in text

class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME_2,
        temperature: float = DEFAULT_TEMPERATURE,
        system_instruction: str = SYSTEM_PROMPT,
    ):
        import google.generativeai as genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY_2")
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key found. Pass api_key= explicitly or set "
                "the GEMINI_API_KEY environment variable."
            )
        genai.configure(api_key=resolved_key)

        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )

    def generate(self, user_prompt: str) -> str:
        try:
            response = self._model.generate_content(
                user_prompt,
                request_options=_NO_SDK_RETRY_REQUEST_OPTIONS,
            )
            return response.text
        except Exception as e:
            if _is_quota_error(e):
                raise QuotaExceededError(str(e), retry_delay_seconds=_extract_retry_delay_seconds(e)) from e
            raise

class GeminiSearchClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        system_instruction: str = RESEARCH_SYSTEM_PROMPT,
    ):
        import google.generativeai as genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key found. Pass api_key= explicitly or set "
                "the GEMINI_API_KEY environment variable."
            )
        genai.configure(api_key=resolved_key)

        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
            tools="google_search_retrieval",
        )

    def generate(self, user_prompt: str) -> str:
        try:
            response = self._model.generate_content(
                user_prompt,
                request_options=_NO_SDK_RETRY_REQUEST_OPTIONS,
            )
            return response.text
        except Exception as e:
            if _is_quota_error(e):
                raise QuotaExceededError(str(e), retry_delay_seconds=_extract_retry_delay_seconds(e)) from e
            raise

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _attempt_json_repair(text: str) -> Optional[str]:
    from json_repair import repair_json

    try:
        candidate = repair_json(text)
    except Exception:
        return None

    if not candidate:
        return None

    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return candidate


def _validate_and_normalize(raw_text: str, expected_vehicle_id: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        repaired = _attempt_json_repair(text)
        if repaired is not None:
            data = json.loads(repaired)
        else:
            raise DecisionParseError(f"invalid JSON: {e}")

    if not isinstance(data, dict):
        raise DecisionParseError("response is not a JSON object")

    if "actions" not in data or not isinstance(data["actions"], list):
        raise DecisionParseError("response missing an 'actions' list")

    if not data["actions"]:
        raise DecisionParseError("'actions' list is empty — model must emit at least no_action")

    returned_vid = data.get("vehicle_id")
    if returned_vid and returned_vid != expected_vehicle_id:
        raise DecisionParseError(
            f"vehicle_id mismatch: expected {expected_vehicle_id}, got {returned_vid}"
        )
    data["vehicle_id"] = expected_vehicle_id

    normalized_actions = []
    for i, action in enumerate(data["actions"]):
        if not isinstance(action, dict):
            raise DecisionParseError(f"action[{i}] is not an object")

        action_type = action.get("action_type")
        if action_type not in ACTION_TYPES:
            raise DecisionParseError(f"action[{i}] has invalid action_type: {action_type!r}")

        priority = action.get("priority")
        if priority not in PRIORITY_LEVELS:
            priority = "low"

        rationale = action.get("rationale")
        if not rationale or not isinstance(rationale, str):
            raise DecisionParseError(f"action[{i}] missing a rationale string")

        normalized_actions.append({
            "action_type": action_type,
            "priority": priority,
            "rationale": rationale,
            "parameters": action.get("parameters") or {},
        })

    data["actions"] = normalized_actions
    data["summary"] = data.get("summary") or "(model did not provide a summary)"
    return data


def _validate_and_normalize_research(raw_text: str, expected_vehicle_id: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        repaired = _attempt_json_repair(text)
        if repaired is not None:
            data = json.loads(repaired)
        else:
            raise DecisionParseError(f"invalid JSON: {e}")

    if not isinstance(data, dict) or "in_scope" not in data:
        raise DecisionParseError("response missing 'in_scope'")

    returned_vid = data.get("vehicle_id")
    if returned_vid and returned_vid != expected_vehicle_id:
        raise DecisionParseError(
            f"vehicle_id mismatch: expected {expected_vehicle_id}, got {returned_vid}"
        )

    return {
        "vehicle_id": expected_vehicle_id,
        "in_scope": bool(data.get("in_scope")),
        "answer": data.get("answer") or "",
        "refusal_reason": data.get("refusal_reason") or "",
    }


def _fallback_decision(vehicle_id: str, error: str) -> Dict[str, Any]:
    return {
        "vehicle_id": vehicle_id,
        "actions": [{
            "action_type": "no_action",
            "priority": "low",
            "rationale": (
                f"Agent reasoning step failed after {MAX_ATTEMPTS} attempt(s) "
                f"({error}). Flagged for manual review rather than silently "
                f"skipped — this is a pipeline failure, not a signal-based decision."
            ),
            "parameters": {"error": error},
        }],
        "summary": f"Automated reasoning failed for {vehicle_id}; needs manual review.",
    }


def _fallback_research(vehicle_id: str, error: str) -> Dict[str, Any]:
    return {
        "vehicle_id": vehicle_id,
        "in_scope": False,
        "answer": "",
        "refusal_reason": f"Research step failed after {MAX_ATTEMPTS} attempt(s) ({error}).",
    }

@dataclass
class DecisionEngine:
    client: GeminiClient
    search_client: Optional[GeminiSearchClient] = None   # None => research disabled
    include_few_shot: bool = True
    retry_backoff_seconds: float = 1.5

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        enable_research: bool = False,
    ) -> "DecisionEngine":
        client = GeminiClient(api_key=api_key, model_name=model_name, temperature=temperature)
        search_client = (
            GeminiSearchClient(api_key=api_key, model_name=model_name, temperature=temperature)
            if enable_research else None
        )
        return cls(client=client, search_client=search_client)

    def decide_for_asset(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        vehicle_id = profile.get("vehicle_id", "UNKNOWN")
        prompt = build_decision_prompt(profile, include_few_shot=self.include_few_shot)

        last_error = "unknown error"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw_text = self.client.generate(prompt)
                return _validate_and_normalize(raw_text, vehicle_id)
            except DecisionParseError as e:
                last_error = str(e)
            except QuotaExceededError as e:
                delay = e.retry_delay_seconds
                hint = f", retry in ~{int(delay)}s" if delay else ""
                last_error = f"quota exceeded{hint} ({e})"
                break
            except Exception as e:
                last_error = f"API error: {e}"

            if attempt < MAX_ATTEMPTS:
                time.sleep(self.retry_backoff_seconds)

        return _fallback_decision(vehicle_id, last_error)

    def decide_for_fleet(
        self,
        conn: sqlite3.Connection,
        risk_engine: Optional[Any] = None,
        execute: bool = True,
    ) -> List[Dict[str, Any]]:
        if risk_engine is None:
            from models.risk_engine import RiskEngine
            risk_engine = RiskEngine()

        profile_df: pd.DataFrame = risk_engine.build_fleet_profile(conn)

        results = []
        for _, row in profile_df.iterrows():
            profile = row.to_dict()
            decision = self.decide_for_asset(profile)

            action_records = []
            if execute:
                action_records = execute_decision(conn, decision)

            results.append({
                "vehicle_id": decision["vehicle_id"],
                "decision": decision,
                "action_records": action_records,
            })

        return results

    def research_asset_question(self, profile: Dict[str, Any], question: str) -> Dict[str, Any]:
        vehicle_id = profile.get("vehicle_id", "UNKNOWN")
        if self.search_client is None:
            return {
                "vehicle_id": vehicle_id,
                "in_scope": False,
                "answer": "",
                "refusal_reason": "Web research is not enabled for this session.",
            }

        prompt = build_research_prompt(profile, question)

        last_error = "unknown error"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw_text = self.search_client.generate(prompt)
                return _validate_and_normalize_research(raw_text, vehicle_id)
            except DecisionParseError as e:
                last_error = str(e)
            except QuotaExceededError as e:
                delay = e.retry_delay_seconds
                hint = f", retry in ~{int(delay)}s" if delay else ""
                last_error = f"quota exceeded{hint} ({e})"
                break
            except Exception as e:
                last_error = f"API error: {e}"

            if attempt < MAX_ATTEMPTS:
                time.sleep(self.retry_backoff_seconds)

        return _fallback_research(vehicle_id, last_error)


if __name__ == "__main__":
    import math

    from ingestion.db import (
        get_connection, init_db, insert_telemetry_batch,
        insert_maintenance_batch, insert_command_batch,
    )
    from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent
    from simulator.config import SimulatorConfig
    from simulator.telemetry_generator import TelemetryGenerator
    from simulator.maintenance_generator import MaintenanceGenerator
    from simulator.attack_injector import AttackInjector
    from models.risk_engine import RiskEngine
    from models.rul_model import RULModel
    from .actions import get_all_actions

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set — skipping live decision_engine sanity check.")
        raise SystemExit(0)

    cfg = SimulatorConfig(
        fleet_size=5, num_cycles=150, random_seed=99,
        attack_injection_rate_pct=0.4,  # bias toward at least one attacked asset for the demo
    )
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    test_db = os.path.join("data", "decision_engine_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**r) for r in tickets_df.to_dict(orient="records")]
    commands = []
    for r in commands_df.to_dict(orient="records"):
        if isinstance(r.get("ticket_id"), float) and math.isnan(r["ticket_id"]):
            r["ticket_id"] = None
        commands.append(CommandEvent(**r))

    insert_telemetry_batch(conn, readings)
    insert_maintenance_batch(conn, tickets)
    insert_command_batch(conn, commands)

    risk_engine = RiskEngine(
        rul_model=RULModel(end_of_life_capacity_pct=cfg.end_of_life_capacity_pct * 100)
    )
    engine = DecisionEngine.create()

    results = engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=True)

    print(f"Ran the agent loop over {len(results)} assets:\n")
    for r in results:
        print(f"--- {r['vehicle_id']} ---")
        print(f"  summary: {r['decision']['summary']}")
        for a in r["decision"]["actions"]:
            print(f"  [{a['priority']:>8}] {a['action_type']}: {a['rationale']}")
        print()

    print("Logged agent_actions rows (excluding no_action):")
    for row in get_all_actions(conn, exclude_no_action=True):
        print(f"  {row['action_id']} | {row['vehicle_id']} | {row['action_type']} | {row['priority']}")

    conn.close()
    os.remove(test_db)