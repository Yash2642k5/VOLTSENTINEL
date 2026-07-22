"""
agent/decision_engine.py

The integration point of the Perceive -> Reason -> Decide -> Act loop
(project doc §6.1). This is what actually makes VoltSentinel an agent
rather than a scoring dashboard.

Wiring, per the build order:
    models/risk_engine.py  -> merged per-asset profile      (Perceive)
    agent/prompts.py       -> system + per-asset prompt text (framing)
    Gemini API             -> reasons over the profile       (Reason/Decide)
    agent/actions.py       -> executes the returned actions  (Act)

Design notes:
  - The LLM is asked for JSON only (response_mime_type="application/json"),
    so we don't need to regex-strip markdown fences in the common case —
    but the parser still defensively strips fences if a stray ``` shows up,
    since providers aren't 100% consistent about honoring JSON mode.
  - Every field the LLM returns is validated against prompts.ACTION_TYPES /
    PRIORITY_LEVELS before it ever reaches actions.py. Invalid enum values
    are coerced to a safe default rather than raising, since a fleet
    manager should never lose visibility into an asset just because the
    model returned "Medium" instead of "medium".
  - If parsing/validation fails even after one retry, we do NOT drop the
    asset silently. We log an explicit no_action decision explaining that
    the agent's reasoning step failed and the asset needs human review —
    keeping the audit trail complete, per §7.3 / §6.1's explainability goal.
  - Only agent/decision_engine.py talks to the network. prompts.py /
    actions.py / risk_engine.py stay provider-agnostic and network-free,
    so swapping Gemini for another provider later only touches this file.

Web search / research addition:
  - GeminiClient (the main decision loop's client) is UNCHANGED and still
    has zero tools registered — decide_for_asset()/decide_for_fleet() stay
    completely air-gapped from the network, so every emitted action still
    traces back only to an already-computed risk_engine.py signal.
  - GeminiSearchClient is a SEPARATE client, built with a different system
    prompt (RESEARCH_SYSTEM_PROMPT) and the only place in this file (or
    the codebase) that registers a tool. It is reachable only through
    DecisionEngine.research_asset_question(), never from the main loop.
  - research_asset_question() is scoped to one asset's already-computed
    profile and relies on RESEARCH_SYSTEM_PROMPT instructing the model to
    refuse anything outside a fixed topic list (see agent/prompts.py).
    Disabled by default — DecisionEngine.create() only builds a
    search_client when enable_research=True is passed explicitly.

Quota / rate-limit handling (free-tier Gemini):
  - The free tier caps generate_content at a small requests-per-minute
    quota per model. Two things used to make a 429 here far worse than it
    needed to be:
      1. google-generativeai's underlying transport has its own automatic
         retry-with-backoff for transient errors, which can silently retry
         for minutes UNDER a single generate_content() call before ever
         raising back to this file. request_options={"retry": None} below
         disables that, so a 429 surfaces immediately.
      2. This file's own retry loops (decide_for_asset,
         research_asset_question) used to retry a 429 with the same fixed
         backoff as a parse error — pointless, since retrying against an
         already-exhausted per-minute quota within the same minute just
         burns the remaining retry budget without succeeding sooner.
    QuotaExceededError below is a distinct exception type so callers (this
    file's retry loops, and agent/tool_chat_engine.py's run_tool_loop) can
    detect a 429 specifically and fail fast — one clear message, using the
    server's own suggested retry_delay when available — instead of
    retrying blindly or hanging for minutes.
"""

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

DEFAULT_MODEL_NAME = "gemini-3.5-flash"
# gemini-2.0-flash was deprecated by Google on 2026-03-03 (scheduled
# shutdown mid-to-late 2026) and its free-tier quota is effectively 0 as
# a result -- every request fails with 429 "limit: 0" regardless of prompt
# size. gemini-3.5-flash is the current-generation free-tier model as of
# mid-2026; re-check https://ai.google.dev/gemini-api/docs/rate-limits
# periodically, since Google has moved this line multiple times in 2026
# (Pro models were removed from the free tier entirely in April 2026). If
# this 429s with "limit: 0" for your specific key, confirm the exact free-
# tier model name in your AI Studio project page and swap it in here.
DEFAULT_TEMPERATURE = 0.2
MAX_ATTEMPTS = 2  # 1 initial call + 1 retry on a parse/validation failure.
# Lowered from 3 -> 2 while on the free tier (5 requests/minute/model):
# every retry is a fresh generate_content() call against the same quota
# bucket, and a QuotaExceededError now skips remaining retries entirely
# (see decide_for_asset below) rather than consuming this budget anyway.
# json_repair-based recovery (_attempt_json_repair below) still handles
# the most common "almost valid JSON" failures without needing a second
# network round trip at all.


class DecisionParseError(Exception):
    """Raised internally when the LLM response can't be turned into a
    valid decision object. Always caught inside decide_for_asset — never
    escapes to the caller, since a single asset's failure shouldn't kill
    a fleet run."""


class QuotaExceededError(Exception):
    """Raised when the underlying Gemini call fails with a 429 (rate/quota
    exceeded) — kept distinct from a generic API/network error so callers
    can fail fast instead of retrying with the same backoff a transient
    error would get. Retrying a per-minute quota error within the same
    minute cannot succeed sooner; it only burns the retry budget and, if
    the SDK's own internal retry is also active, can silently stall for
    minutes before ever raising. See module docstring."""

    def __init__(self, message: str, retry_delay_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds


# Disables google-generativeai's own internal transport-level retry/backoff
# for a single generate_content() call, so a 429 raises immediately instead
# of being silently retried (with exponential backoff) inside the SDK for
# up to several minutes before ever reaching our code. We do our own,
# explicit, fail-fast handling instead (see QuotaExceededError above).
_NO_SDK_RETRY_REQUEST_OPTIONS = {"retry": None}


def _extract_retry_delay_seconds(error: Exception) -> Optional[float]:
    """Best-effort extraction of the server-suggested retry delay (seconds)
    from a 429/ResourceExhausted error. The google-api-core exception's
    structured fields for this vary across SDK versions, so this falls
    back to a regex over str(error) — the free-tier 429 body has
    consistently included a `retry_delay { seconds: N }` block in testing.
    Returns None if no delay could be determined; callers must handle that."""
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(error))
    if match:
        return float(match.group(1))
    # Fall back to any bare "seconds: N" in case the wrapping changes.
    match = re.search(r"seconds:\s*(\d+)", str(error))
    if match:
        return float(match.group(1))
    return None


def _is_quota_error(error: Exception) -> bool:
    """Detects a 429/quota-exceeded error without a hard dependency on a
    specific google-api-core exception class (import paths for
    ResourceExhausted have moved across SDK versions). Checking both the
    class name and message text keeps this robust to that churn."""
    if type(error).__name__ in ("ResourceExhausted", "TooManyRequests"):
        return True
    text = str(error)
    return "429" in text or "quota" in text.lower() or "RESOURCE_EXHAUSTED" in text


# ----------------------------------------------------------------------
# Gemini client wrapper — isolated so tests can monkeypatch/mock it
# without touching parsing/validation/dispatch logic. Deliberately has
# NO tools registered — the main decision loop must stay network-free
# beyond the single Gemini reasoning call itself, per this file's
# explainability guarantee (every action traces back to a risk_engine.py
# signal, never to something fetched mid-reasoning).
# ----------------------------------------------------------------------
class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        system_instruction: str = SYSTEM_PROMPT,
    ):
        # system_instruction defaults to the main decide_for_asset/decide_for_fleet
        # prompt (SYSTEM_PROMPT) so every EXISTING caller — including
        # DecisionEngine.create() below and every test that constructs a
        # GeminiClient positionally/by keyword without this arg — behaves
        # exactly as before. agent/bi_chat_engine.py's BIChatEngine.create()
        # passes system_instruction=BI_SYSTEM_PROMPT explicitly, reusing this
        # same class with the BI-chat framing instead of the decision-loop one.
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


# ----------------------------------------------------------------------
# Search-enabled client — kept entirely separate from GeminiClient above.
# This is the ONLY client construction in the codebase that registers a
# tool (Google Search grounding), and it's built with RESEARCH_SYSTEM_PROMPT
# instead of SYSTEM_PROMPT, so it physically cannot be substituted into the
# main decision loop by accident — the two prompts encode different
# contracts (JSON action schema vs. in_scope/answer/refusal schema).
# ----------------------------------------------------------------------
class GeminiSearchClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        system_instruction: str = RESEARCH_SYSTEM_PROMPT,
    ):
        # system_instruction defaults to RESEARCH_SYSTEM_PROMPT so
        # DecisionEngine.create(enable_research=True)'s existing construction
        # is unchanged. agent/bi_tools.py's web_search tool passes
        # BI_WEB_SEARCH_SYSTEM_PROMPT instead — a much simpler schema (just
        # {"answer", "sources"}) since scope-gating there is the calling
        # model's job (BI_SYSTEM_PROMPT), not this client's.
        import google.generativeai as genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key found. Pass api_key= explicitly or set "
                "the GEMINI_API_KEY environment variable."
            )
        genai.configure(api_key=resolved_key)

        # "google_search_retrieval" is the SDK's built-in search-grounding
        # tool shorthand. A raw dict like {"google_search": {}} is NOT a
        # recognized shorthand -- the SDK instead tries to parse it as a
        # custom FunctionDeclaration and raises "Unknown field for
        # FunctionDeclaration: google_search". If this still fails on your
        # installed google-generativeai version, check that package's
        # current docs for the exact grounding-tool name it expects.
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


# ----------------------------------------------------------------------
# Response validation — keeps bad LLM output from ever reaching actions.py
# ----------------------------------------------------------------------
def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _attempt_json_repair(text: str) -> Optional[str]:
    """Best-effort repair for the most common way Gemini's JSON-mode output
    still comes back malformed despite response_mime_type="application/json"
    (that setting is best-effort, not a hard schema guarantee): a string
    value — almost always a `rationale` that quotes a coordinate or phrase —
    contains a raw, unescaped double-quote or newline, which breaks the
    parser mid-string with an "Expecting ',' delimiter" error partway
    through the document.

    Delegates to the json-repair library rather than a hand-rolled regex
    fix. A naive "insert one comma at the parser's failure position" repair
    was tried first and FAILED on real malformed output — the text
    following an unescaped quote (e.g. `in transit" and no matching
    ticket.",`) usually isn't valid JSON structure no matter where a single
    comma goes, since the actual bug is a missing pair of escape
    characters, not a missing delimiter. json-repair re-scans for the
    intended string boundary and re-escapes correctly instead of patching
    the symptom.

    Returns a repaired string that DOES parse if a fix was found, else
    None — never returns something that still fails to parse, so callers
    can trust a non-None result outright without a second try/except.
    """
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
            data = json.loads(repaired)  # already verified to parse in _attempt_json_repair
        else:
            raise DecisionParseError(f"invalid JSON: {e}")

    if not isinstance(data, dict):
        raise DecisionParseError("response is not a JSON object")

    if "actions" not in data or not isinstance(data["actions"], list):
        raise DecisionParseError("response missing an 'actions' list")

    if not data["actions"]:
        raise DecisionParseError("'actions' list is empty — model must emit at least no_action")

    # vehicle_id: trust our own input over whatever the model echoed back,
    # but require it be present and non-contradictory as a sanity check.
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
            # Don't fail the whole asset over a bad enum value — coerce to a
            # conservative default and keep going; still auditable via rationale.
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
    """Validator for GeminiSearchClient responses — a completely different,
    much smaller schema than _validate_and_normalize above (no actions list,
    no priority/action_type enums). Kept as a separate function rather than
    branching inside _validate_and_normalize so the two contracts can never
    accidentally cross-validate each other's output."""
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
    """Used only when every attempt to get a valid decision from the LLM
    fails. Still a real, logged decision — never a silent drop — so the
    fleet view shows 'agent reasoning failed for this asset, needs human
    review' instead of the asset just vanishing from the audit trail."""
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
    """Mirrors _fallback_decision's contract for the research path — never
    raises out of research_asset_question(), always returns a well-shaped
    refusal instead."""
    return {
        "vehicle_id": vehicle_id,
        "in_scope": False,
        "answer": "",
        "refusal_reason": f"Research step failed after {MAX_ATTEMPTS} attempt(s) ({error}).",
    }


# ----------------------------------------------------------------------
# Decision engine
# ----------------------------------------------------------------------
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
        """enable_research=False by default — web search stays off unless a
        caller explicitly opts in. Even when on, it only ever populates
        search_client, which is reachable exclusively through
        research_asset_question() below; decide_for_asset()/decide_for_fleet()
        never touch it."""
        client = GeminiClient(api_key=api_key, model_name=model_name, temperature=temperature)
        search_client = (
            GeminiSearchClient(api_key=api_key, model_name=model_name, temperature=temperature)
            if enable_research else None
        )
        return cls(client=client, search_client=search_client)

    # ------------------------------------------------------------------
    def decide_for_asset(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Runs Reason -> Decide for a single asset's merged profile
        (one row of risk_engine.build_fleet_profile(), as a dict).
        Always returns a valid decision object matching prompts.py's
        schema — falls back to an explicit review-needed no_action
        rather than raising, so a fleet loop never aborts on one asset."""
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
                # A per-minute quota error will not resolve itself within
                # this function's short retry window, and retrying
                # immediately just burns the rest of the attempt budget for
                # no benefit. Fail fast with the server's own suggested
                # wait time (when available) instead of looping.
                delay = e.retry_delay_seconds
                hint = f", retry in ~{int(delay)}s" if delay else ""
                last_error = f"quota exceeded{hint} ({e})"
                break
            except Exception as e:  # network/API errors — retry the same way
                last_error = f"API error: {e}"

            if attempt < MAX_ATTEMPTS:
                time.sleep(self.retry_backoff_seconds)

        return _fallback_decision(vehicle_id, last_error)

    # ------------------------------------------------------------------
    def decide_for_fleet(
        self,
        conn: sqlite3.Connection,
        risk_engine: Optional[Any] = None,
        execute: bool = True,
    ) -> List[Dict[str, Any]]:
        """Full fleet run: builds the merged risk profile (Perceive),
        reasons over every asset (Reason/Decide), and — unless
        execute=False — dispatches each asset's actions through
        agent/actions.py (Act). Returns one summary dict per asset:
        {"vehicle_id", "decision", "action_records"}."""
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

    # ------------------------------------------------------------------
    # Scoped web research — the ONLY code path in this system with network
    # access beyond the single reasoning call. See module docstring.
    # ------------------------------------------------------------------
    def research_asset_question(self, profile: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Answers a free-text question about ONE asset, using web search,
        but only within the topic list RESEARCH_SYSTEM_PROMPT enforces
        (agent/prompts.py) — e.g. replacement vehicle/battery models,
        charge-policy best practices, or context tied to a signal already
        present in `profile`. The model is instructed to set
        in_scope=False and explain why for anything else (general
        knowledge, unrelated topics, requests to act on other systems);
        this method trusts and passes through that refusal rather than
        second-guessing it.

        Always returns a well-shaped dict — never raises — matching
        decide_for_asset's fail-safe contract. If research isn't enabled
        for this engine instance (self.search_client is None), returns an
        explicit "not enabled" refusal rather than silently doing nothing.
        """
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
                # Same fail-fast reasoning as decide_for_asset above.
                delay = e.retry_delay_seconds
                hint = f", retry in ~{int(delay)}s" if delay else ""
                last_error = f"quota exceeded{hint} ({e})"
                break
            except Exception as e:  # network/API errors — retry the same way
                last_error = f"API error: {e}"

            if attempt < MAX_ATTEMPTS:
                time.sleep(self.retry_backoff_seconds)

        return _fallback_research(vehicle_id, last_error)


if __name__ == "__main__":
    # Standalone sanity check: seed a scratch DB via the simulator ->
    # risk_engine pipeline, then run the full agent loop against it.
    # Requires a real GEMINI_API_KEY in the environment — skips cleanly
    # (rather than failing) if one isn't set, so this file can still be
    # imported/tested in CI without network access or a key.
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
    engine = DecisionEngine.create()  # reads GEMINI_API_KEY from env

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