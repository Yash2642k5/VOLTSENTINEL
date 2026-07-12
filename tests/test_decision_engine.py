"""
tests/test_decision_engine.py

Validates agent/decision_engine.py: response validation/normalization,
the retry -> fallback path when the LLM returns something unusable, and
the full decide_for_fleet loop against a real (simulator-seeded) DB.

No network calls and no GEMINI_API_KEY needed — DecisionEngine takes its
client as a plain object with a .generate(prompt) -> str method, so tests
inject a FakeGeminiClient instead of the real GeminiClient (which is the
only place that imports google.generativeai / touches the network).

Run from the project root:
    pytest tests/test_decision_engine.py -v
"""

import json
import math
import os
import re

import pytest

from agent.decision_engine import (
    DecisionEngine,
    DecisionParseError,
    MAX_ATTEMPTS,
    _fallback_decision,
    _strip_code_fence,
    _validate_and_normalize,
)
from agent.actions import get_all_actions
from agent.prompts import ACTION_TYPES, PRIORITY_LEVELS

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector

from ingestion.db import (
    get_connection, init_db, insert_telemetry_batch,
    insert_maintenance_batch, insert_command_batch,
)
from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent

from models.risk_engine import RiskEngine
from models.rul_model import RULModel


TEST_DB_PATH = os.path.join("data", "test_decision_engine.db")


# ----------------------------------------------------------------------
# A minimal, well-formed decision — reused as the baseline "good" fixture
# ----------------------------------------------------------------------
def _valid_response_text(vehicle_id: str = "EVR-0001") -> str:
    return json.dumps({
        "vehicle_id": vehicle_id,
        "actions": [
            {
                "action_type": "maintenance_trigger",
                "priority": "medium",
                "rationale": "RUL degraded at 76% capacity with recurring thermal anomalies.",
                "parameters": {"reason": "degraded RUL"},
            },
        ],
        "summary": f"{vehicle_id} needs a scheduled inspection soon.",
    })


# ----------------------------------------------------------------------
# Fake Gemini client — queue-based, so tests can script exact call-by-call
# behaviour (e.g. "fail once, then succeed") without any network access.
# ----------------------------------------------------------------------
class FakeGeminiClient:
    """responses: list of items, one per call to .generate(). Each item is
    either a string (returned as-is) or an Exception instance (raised)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.prompts_seen = []

    def generate(self, user_prompt: str) -> str:
        self.call_count += 1
        self.prompts_seen.append(user_prompt)
        if not self._responses:
            raise AssertionError("FakeGeminiClient.generate() called more times than scripted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DynamicFakeGeminiClient:
    """Extracts the vehicle_id from the prompt text (format_asset_profile
    always emits 'Asset: <vehicle_id>') and echoes back a valid no_action
    decision for whichever asset it's asked about — used for fleet-level
    tests where many different vehicle_ids are in play.

    build_decision_prompt() prepends a few-shot example by default, and
    that example text also contains an 'Asset: EVR-0042' line — so we
    must take the LAST match (the real asset, after the '...Output:'
    marker), not the first, or every call ends up echoing the few-shot
    vehicle_id back and tripping decision_engine's own mismatch guard."""

    def __init__(self):
        self.call_count = 0

    def generate(self, user_prompt: str) -> str:
        self.call_count += 1
        matches = re.findall(r"Asset:\s*(\S+)", user_prompt)
        vehicle_id = matches[-1] if matches else "UNKNOWN"
        return json.dumps({
            "vehicle_id": vehicle_id,
            "actions": [{
                "action_type": "no_action",
                "priority": "low",
                "rationale": "All signals nominal for this asset in the test fixture.",
                "parameters": {},
            }],
            "summary": f"{vehicle_id} is nominal.",
        })


# ----------------------------------------------------------------------
# _strip_code_fence
# ----------------------------------------------------------------------
class TestStripCodeFence:
    def test_plain_json_passthrough(self):
        assert _strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _strip_code_fence(text) == '{"a": 1}'

    def test_strips_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _strip_code_fence(text) == '{"a": 1}'


# ----------------------------------------------------------------------
# _validate_and_normalize — the core correctness surface
# ----------------------------------------------------------------------
class TestValidateAndNormalize:
    def test_valid_response_parses(self):
        data = _validate_and_normalize(_valid_response_text("EVR-0001"), "EVR-0001")
        assert data["vehicle_id"] == "EVR-0001"
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action_type"] == "maintenance_trigger"

    def test_invalid_json_raises(self):
        with pytest.raises(DecisionParseError):
            _validate_and_normalize("not json at all", "EVR-0001")

    def test_non_object_json_raises(self):
        with pytest.raises(DecisionParseError):
            _validate_and_normalize("[1, 2, 3]", "EVR-0001")

    def test_missing_actions_key_raises(self):
        with pytest.raises(DecisionParseError):
            _validate_and_normalize(json.dumps({"vehicle_id": "EVR-0001"}), "EVR-0001")

    def test_empty_actions_list_raises(self):
        with pytest.raises(DecisionParseError):
            _validate_and_normalize(
                json.dumps({"vehicle_id": "EVR-0001", "actions": []}), "EVR-0001"
            )

    def test_vehicle_id_mismatch_raises(self):
        with pytest.raises(DecisionParseError):
            _validate_and_normalize(_valid_response_text("EVR-9999"), "EVR-0001")

    def test_missing_vehicle_id_is_filled_in(self):
        raw = json.dumps({
            "actions": [{
                "action_type": "no_action", "priority": "low",
                "rationale": "nothing to do", "parameters": {},
            }],
        })
        data = _validate_and_normalize(raw, "EVR-0001")
        assert data["vehicle_id"] == "EVR-0001"

    def test_invalid_action_type_raises(self):
        raw = json.dumps({
            "vehicle_id": "EVR-0001",
            "actions": [{
                "action_type": "reboot_vehicle",  # not in ACTION_TYPES
                "priority": "low", "rationale": "x", "parameters": {},
            }],
        })
        with pytest.raises(DecisionParseError):
            _validate_and_normalize(raw, "EVR-0001")

    def test_invalid_priority_is_coerced_not_raised(self):
        raw = json.dumps({
            "vehicle_id": "EVR-0001",
            "actions": [{
                "action_type": "no_action",
                "priority": "URGENT!!!",  # not in PRIORITY_LEVELS
                "rationale": "x", "parameters": {},
            }],
        })
        data = _validate_and_normalize(raw, "EVR-0001")
        assert data["actions"][0]["priority"] == "low"

    def test_missing_rationale_raises(self):
        raw = json.dumps({
            "vehicle_id": "EVR-0001",
            "actions": [{"action_type": "no_action", "priority": "low", "parameters": {}}],
        })
        with pytest.raises(DecisionParseError):
            _validate_and_normalize(raw, "EVR-0001")

    def test_missing_summary_gets_default(self):
        raw = json.dumps({
            "vehicle_id": "EVR-0001",
            "actions": [{
                "action_type": "no_action", "priority": "low",
                "rationale": "nothing to do", "parameters": {},
            }],
        })
        data = _validate_and_normalize(raw, "EVR-0001")
        assert data["summary"]

    def test_missing_parameters_defaults_to_empty_dict(self):
        raw = json.dumps({
            "vehicle_id": "EVR-0001",
            "actions": [{"action_type": "no_action", "priority": "low", "rationale": "x"}],
        })
        data = _validate_and_normalize(raw, "EVR-0001")
        assert data["actions"][0]["parameters"] == {}

    def test_code_fenced_response_parses(self):
        fenced = f"```json\n{_valid_response_text('EVR-0001')}\n```"
        data = _validate_and_normalize(fenced, "EVR-0001")
        assert data["vehicle_id"] == "EVR-0001"

    def test_all_action_types_individually_accepted(self):
        for action_type in ACTION_TYPES:
            raw = json.dumps({
                "vehicle_id": "EVR-0001",
                "actions": [{
                    "action_type": action_type, "priority": "low",
                    "rationale": "exercising every action type", "parameters": {},
                }],
            })
            data = _validate_and_normalize(raw, "EVR-0001")
            assert data["actions"][0]["action_type"] == action_type

    def test_all_priority_levels_individually_accepted(self):
        for priority in PRIORITY_LEVELS:
            raw = json.dumps({
                "vehicle_id": "EVR-0001",
                "actions": [{
                    "action_type": "no_action", "priority": priority,
                    "rationale": "exercising every priority level", "parameters": {},
                }],
            })
            data = _validate_and_normalize(raw, "EVR-0001")
            assert data["actions"][0]["priority"] == priority


# ----------------------------------------------------------------------
# _fallback_decision
# ----------------------------------------------------------------------
class TestFallbackDecision:
    def test_fallback_is_itself_a_valid_shape(self):
        """The fallback must pass the exact same validator real responses
        do, since actions.execute_decision() doesn't know it's a fallback."""
        decision = _fallback_decision("EVR-0001", "boom")
        reserialized = _validate_and_normalize(json.dumps(decision), "EVR-0001")
        assert reserialized["actions"][0]["action_type"] == "no_action"

    def test_fallback_rationale_mentions_the_error(self):
        decision = _fallback_decision("EVR-0001", "invalid JSON: boom")
        assert "boom" in decision["actions"][0]["rationale"]


# ----------------------------------------------------------------------
# DecisionEngine.decide_for_asset — retry / fallback behaviour
# ----------------------------------------------------------------------
class TestDecideForAsset:
    def _engine(self, responses):
        engine = DecisionEngine(client=FakeGeminiClient(responses))
        engine.retry_backoff_seconds = 0  # don't slow down the test suite
        return engine

    def test_succeeds_on_first_try(self):
        engine = self._engine([_valid_response_text("EVR-0001")])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0001"})
        assert decision["vehicle_id"] == "EVR-0001"
        assert engine.client.call_count == 1

    def test_retries_once_then_succeeds(self):
        engine = self._engine(["not valid json", _valid_response_text("EVR-0001")])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0001"})
        assert decision["vehicle_id"] == "EVR-0001"
        assert decision["actions"][0]["action_type"] == "maintenance_trigger"
        assert engine.client.call_count == 2

    def test_falls_back_after_exhausting_retries(self):
        engine = self._engine(["garbage 1", "garbage 2", "garbage 3"])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0001"})
        assert decision["actions"][0]["action_type"] == "no_action"
        assert "review" in decision["actions"][0]["rationale"].lower()
        assert engine.client.call_count == MAX_ATTEMPTS  # never exceeds the configured cap

    def test_api_exception_triggers_retry_not_crash(self):
        engine = self._engine([RuntimeError("connection reset"), _valid_response_text("EVR-0001")])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0001"})
        assert decision["vehicle_id"] == "EVR-0001"
        assert engine.client.call_count == 2

    def test_never_raises_even_on_total_failure(self):
        """decide_for_asset must never propagate an exception — a fleet
        loop should never abort because one asset's LLM call kept failing."""
        engine = self._engine([RuntimeError("down"), RuntimeError("still down")])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0001"})
        assert decision["actions"][0]["action_type"] == "no_action"

    def test_prompt_includes_the_vehicle_id(self):
        engine = self._engine([_valid_response_text("EVR-0007")])
        engine.decide_for_asset({"vehicle_id": "EVR-0007", "status": "healthy"})
        assert "EVR-0007" in engine.client.prompts_seen[0]

    def test_wrong_vehicle_id_from_model_never_gets_misattributed(self):
        """If the model ever echoes back a different asset's ID (e.g.
        confused by the few-shot example, which itself references
        'EVR-0042'), decide_for_asset must NOT silently attribute that
        action to the asset we actually asked about — it should exhaust
        retries and fall back to an explicit review-needed decision
        instead of mislabeling which vehicle an action belongs to."""
        engine = self._engine([
            _valid_response_text("EVR-0042"),  # wrong asset, both attempts
            _valid_response_text("EVR-0042"),
        ])
        decision = engine.decide_for_asset({"vehicle_id": "EVR-0007"})
        assert decision["vehicle_id"] == "EVR-0007"
        assert decision["actions"][0]["action_type"] == "no_action"
        assert engine.client.call_count == MAX_ATTEMPTS


# ----------------------------------------------------------------------
# DecisionEngine.decide_for_fleet — full loop against a real seeded DB
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(
        fleet_size=6, num_cycles=120, random_seed=4321,
        attack_injection_rate_pct=0.3,
    )


@pytest.fixture(scope="module")
def simulated_data(small_config):
    tgen = TelemetryGenerator(small_config)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(small_config)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(small_config)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)
    return telem_df, tickets_df, commands_df


@pytest.fixture
def conn(small_config, simulated_data):
    telem_df, tickets_df, commands_df = simulated_data
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**r) for r in tickets_df.to_dict(orient="records")]
    commands = []
    for r in commands_df.to_dict(orient="records"):
        if isinstance(r.get("ticket_id"), float) and math.isnan(r["ticket_id"]):
            r["ticket_id"] = None
        commands.append(CommandEvent(**r))

    insert_telemetry_batch(connection, readings)
    insert_maintenance_batch(connection, tickets)
    insert_command_batch(connection, commands)

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def risk_engine(small_config):
    return RiskEngine(
        rul_model=RULModel(end_of_life_capacity_pct=small_config.end_of_life_capacity_pct * 100)
    )


class TestDecideForFleet:
    def test_every_vehicle_gets_a_decision(self, small_config, conn, risk_engine):
        engine = DecisionEngine(client=DynamicFakeGeminiClient())
        results = engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=False)
        assert len(results) == small_config.fleet_size
        vehicle_ids = {r["vehicle_id"] for r in results}
        assert len(vehicle_ids) == small_config.fleet_size  # no duplicates, none dropped

    def test_each_decision_matches_the_asset_it_was_built_for(self, conn, risk_engine):
        """Regression guard: catches any accidental index/order mismatch
        between the risk profile rows and the decisions returned for them
        (the same class of bug test_anomaly_detector.py guards against)."""
        engine = DecisionEngine(client=DynamicFakeGeminiClient())
        results = engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=False)
        for r in results:
            assert r["decision"]["vehicle_id"] == r["vehicle_id"]

    def test_execute_true_writes_audit_rows(self, conn, risk_engine):
        engine = DecisionEngine(client=DynamicFakeGeminiClient())
        results = engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=True)
        for r in results:
            assert len(r["action_records"]) >= 1

        logged = get_all_actions(conn)
        # DynamicFakeGeminiClient always emits exactly one no_action per asset
        assert len(logged) == len(results)

    def test_execute_false_writes_nothing(self, small_config, conn, risk_engine):
        engine = DecisionEngine(client=DynamicFakeGeminiClient())
        results = engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=False)
        for r in results:
            assert r["action_records"] == []
        assert len(get_all_actions(conn)) == 0

    def test_one_llm_call_per_asset(self, small_config, conn, risk_engine):
        client = DynamicFakeGeminiClient()
        engine = DecisionEngine(client=client)
        engine.decide_for_fleet(conn, risk_engine=risk_engine, execute=False)
        assert client.call_count == small_config.fleet_size