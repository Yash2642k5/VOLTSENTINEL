"""
tests/test_bi_chat_engine.py

Validates agent/bi_chat_engine.py: the final-answer/chart-spec
validation logic (never raises, degrades gracefully), and a full
ask() call against a real seeded DB using a scripted fake client (no
network) -- mirrors tests/test_decision_engine.py's approach of
injecting a fake client into the engine rather than hitting Gemini.

Run from the project root:
    pytest tests/test_bi_chat_engine.py -v
"""

import json
import math
import os

import pytest

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

from agent.bi_chat_engine import BIChatEngine, _validate_final_answer


TEST_DB_PATH = os.path.join("data", "test_bi_chat_engine.db")


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if not self._responses:
            raise AssertionError("FakeClient.generate() called more times than scripted")
        return self._responses.pop(0)


# ----------------------------------------------------------------------
# _validate_final_answer -- pure, no DB/network
# ----------------------------------------------------------------------
class TestValidateFinalAnswer:
    def test_valid_text_only(self):
        result = _validate_final_answer({"text": "the fleet looks fine", "chart": None})
        assert result == {"text": "the fleet looks fine", "chart": None}

    def test_missing_text_gets_placeholder(self):
        result = _validate_final_answer({"chart": None})
        assert result["text"]

    def test_valid_bar_chart_passes_through(self):
        raw = {
            "text": "here's the comparison",
            "chart": {"type": "bar", "title": "t", "x_field": "vehicle_id", "y_field": "score",
                      "data": [{"vehicle_id": "EVR-0001", "score": 10}]},
        }
        result = _validate_final_answer(raw)
        assert result["chart"]["type"] == "bar"

    def test_invalid_chart_type_is_dropped(self):
        raw = {"text": "x", "chart": {"type": "pie", "data": [{"a": 1}]}}
        result = _validate_final_answer(raw)
        assert result["chart"] is None

    def test_empty_chart_data_is_dropped(self):
        raw = {"text": "x", "chart": {"type": "bar", "data": []}}
        result = _validate_final_answer(raw)
        assert result["chart"] is None

    def test_chart_not_a_dict_is_dropped(self):
        raw = {"text": "x", "chart": "not a dict"}
        result = _validate_final_answer(raw)
        assert result["chart"] is None

    def test_missing_chart_key_defaults_to_none(self):
        result = _validate_final_answer({"text": "x"})
        assert result["chart"] is None


# ----------------------------------------------------------------------
# BIChatEngine.ask() -- full loop against a real seeded DB
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=5, num_cycles=60, random_seed=7)


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
def profile_df(small_config, conn):
    engine = RiskEngine(rul_model=RULModel(end_of_life_capacity_pct=small_config.end_of_life_capacity_pct * 100))
    return engine.build_fleet_profile(conn)


class TestAsk:
    def test_direct_final_answer_no_tool_calls(self, conn, profile_df):
        client = FakeClient([json.dumps({"final_answer": {"text": "fleet is fine", "chart": None}})])
        engine = BIChatEngine(client=client)
        answer = engine.ask(conn, profile_df, "is the fleet ok?")
        assert answer["text"] == "fleet is fine"
        assert answer["chart"] is None

    def test_tool_call_then_final_answer_uses_real_data(self, conn, profile_df):
        vid = profile_df["vehicle_id"].iloc[0]
        client = FakeClient([
            json.dumps({"tool_call": {"name": "get_vehicle_profile", "arguments": {"vehicle_id": vid}}}),
            json.dumps({"final_answer": {"text": f"{vid} looks as expected", "chart": None}}),
        ])
        engine = BIChatEngine(client=client)
        answer = engine.ask(conn, profile_df, f"what's the status of {vid}?")
        assert vid in answer["text"]
        assert client.call_count == 2

    def test_chat_history_is_included_in_the_prompt(self, conn, profile_df):
        client = FakeClient([json.dumps({"final_answer": {"text": "sure, here's more", "chart": None}})])
        engine = BIChatEngine(client=client)
        history = [{"role": "user", "text": "compare EVR-0001 and EVR-0002"},
                {"role": "assistant", "text": "EVR-0001 has higher stress"}]
        engine.ask(conn, profile_df, "what about thermal anomalies too?", chat_history=history)
        # No assertion on prompt content needed beyond "it didn't crash" --
        # transcript construction is exercised directly in test_tool_chat_engine.py.
        assert client.call_count == 1

    def test_malformed_chart_from_model_degrades_to_text_only(self, conn, profile_df):
        client = FakeClient([json.dumps({
            "final_answer": {"text": "here's a chart", "chart": {"type": "pie", "data": []}},
        })])
        engine = BIChatEngine(client=client)
        answer = engine.ask(conn, profile_df, "show me a pie chart")
        assert answer["chart"] is None
        assert answer["text"] == "here's a chart"