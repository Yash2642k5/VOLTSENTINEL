"""
tests/test_bi_tools.py

Validates agent/bi_tools.py's read-only query tools against a real
simulator-seeded DB + risk profile — no LLM/network involved, matching
this project's existing test style of exercising the pure logic layer
directly (tests/test_risk_engine.py, tests/test_rul_model.py, etc.).

Run from the project root:
    pytest tests/test_bi_tools.py -v
"""

import math
import os

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector
from simulator.asset_generator import AssetGenerator

from ingestion.db import (
    get_connection, init_db, insert_telemetry_batch,
    insert_maintenance_batch, insert_command_batch, insert_vehicle_metadata_batch,
)
from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent, VehicleMetadata

from models.risk_engine import RiskEngine
from models.rul_model import RULModel

from agent.bi_tools import build_bi_tools, RANKABLE_METRICS, TIMESERIES_METRICS


TEST_DB_PATH = os.path.join("data", "test_bi_tools.db")


@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=6, num_cycles=80, random_seed=99, attack_injection_rate_pct=0.3)


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


@pytest.fixture(scope="module")
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

    agen = AssetGenerator(small_config)
    tgen_for_ids = TelemetryGenerator(small_config)
    assets_df = agen.generate_fleet_assets(tgen_for_ids.vehicle_ids)
    assets = [VehicleMetadata(**r) for r in assets_df.to_dict(orient="records")]
    insert_vehicle_metadata_batch(connection, assets)

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="module")
def profile_df(small_config, conn):
    engine = RiskEngine(rul_model=RULModel(end_of_life_capacity_pct=small_config.end_of_life_capacity_pct * 100))
    return engine.build_fleet_profile(conn)


@pytest.fixture(scope="module")
def tools(conn, profile_df):
    return build_bi_tools(conn, profile_df)


# ----------------------------------------------------------------------
# get_fleet_summary
# ----------------------------------------------------------------------
class TestGetFleetSummary:
    def test_total_matches_fleet_size(self, small_config, tools):
        result = tools["get_fleet_summary"].fn()
        assert result["total_vehicles"] == small_config.fleet_size

    def test_risk_counts_sum_to_total(self, tools):
        result = tools["get_fleet_summary"].fn()
        assert sum(result["risk_level_counts"].values()) == result["total_vehicles"]

    def test_empty_profile_returns_zero(self, conn):
        empty_tools = build_bi_tools(conn, pd.DataFrame())
        result = empty_tools["get_fleet_summary"].fn()
        assert result["total_vehicles"] == 0


# ----------------------------------------------------------------------
# list_vehicles
# ----------------------------------------------------------------------
class TestListVehicles:
    def test_no_filter_returns_all(self, small_config, tools):
        result = tools["list_vehicles"].fn()
        assert result["matched"] == small_config.fleet_size

    def test_filter_by_risk_level_narrows_results(self, profile_df, tools):
        any_level = profile_df["overall_risk_level"].iloc[0]
        result = tools["list_vehicles"].fn(risk_level=any_level)
        assert result["matched"] == int((profile_df["overall_risk_level"] == any_level).sum())
        assert all(v["overall_risk_level"] == any_level for v in result["vehicles"])

    def test_limit_caps_returned_rows(self, tools):
        result = tools["list_vehicles"].fn(limit=1)
        assert len(result["vehicles"]) <= 1

    def test_unmatched_filter_returns_empty_not_error(self, tools):
        result = tools["list_vehicles"].fn(status="not_a_real_status")
        assert result["vehicles"] == []
        assert result["matched"] == 0


# ----------------------------------------------------------------------
# get_vehicle_profile / compare_vehicles
# ----------------------------------------------------------------------
class TestVehicleLookup:
    def test_get_vehicle_profile_known_id(self, profile_df, tools):
        vid = profile_df["vehicle_id"].iloc[0]
        result = tools["get_vehicle_profile"].fn(vehicle_id=vid)
        assert result["vehicle_id"] == vid
        assert "error" not in result

    def test_get_vehicle_profile_unknown_id_returns_error_not_raise(self, tools):
        result = tools["get_vehicle_profile"].fn(vehicle_id="EVR-9999")
        assert "error" in result

    def test_compare_vehicles_multiple_known_ids(self, profile_df, tools):
        ids = list(profile_df["vehicle_id"].iloc[:2])
        result = tools["compare_vehicles"].fn(vehicle_ids=",".join(ids))
        assert len(result["vehicles"]) == 2
        assert "not_found" not in result

    def test_compare_vehicles_mixed_known_and_unknown(self, profile_df, tools):
        vid = profile_df["vehicle_id"].iloc[0]
        result = tools["compare_vehicles"].fn(vehicle_ids=f"{vid},EVR-9999")
        assert len(result["vehicles"]) == 1
        assert result["not_found"] == ["EVR-9999"]

    def test_compare_vehicles_empty_string_returns_error(self, tools):
        result = tools["compare_vehicles"].fn(vehicle_ids="")
        assert "error" in result


# ----------------------------------------------------------------------
# rank_vehicles
# ----------------------------------------------------------------------
class TestRankVehicles:
    def test_unknown_metric_returns_error(self, tools):
        result = tools["rank_vehicles"].fn(metric="not_a_real_metric")
        assert "error" in result

    def test_every_rankable_metric_works(self, tools):
        for metric in RANKABLE_METRICS:
            result = tools["rank_vehicles"].fn(metric=metric, limit=5)
            assert "error" not in result

    def test_ascending_actually_sorts_ascending(self, tools):
        result = tools["rank_vehicles"].fn(metric="charge_stress_score", ascending=True, limit=50)
        scores = [v["charge_stress_score"] for v in result["vehicles"] if v["charge_stress_score"] is not None]
        assert scores == sorted(scores)

    def test_limit_is_respected(self, tools):
        result = tools["rank_vehicles"].fn(metric="rul_cycles", limit=2)
        assert len(result["vehicles"]) <= 2


# ----------------------------------------------------------------------
# get_vehicle_timeseries / compare_vehicle_timeseries
# ----------------------------------------------------------------------
class TestTimeseries:
    def test_unknown_metric_returns_error(self, profile_df, tools):
        vid = profile_df["vehicle_id"].iloc[0]
        result = tools["get_vehicle_timeseries"].fn(vehicle_id=vid, metric="not_a_metric")
        assert "error" in result

    def test_every_timeseries_metric_works(self, profile_df, tools):
        vid = profile_df["vehicle_id"].iloc[0]
        for metric in TIMESERIES_METRICS:
            result = tools["get_vehicle_timeseries"].fn(vehicle_id=vid, metric=metric)
            assert "error" not in result
            assert len(result["points"]) > 0

    def test_unknown_vehicle_returns_error(self, tools):
        result = tools["get_vehicle_timeseries"].fn(vehicle_id="EVR-9999", metric="temperature_c")
        assert "error" in result

    def test_compare_timeseries_multiple_vehicles(self, profile_df, tools):
        ids = list(profile_df["vehicle_id"].iloc[:2])
        result = tools["compare_vehicle_timeseries"].fn(vehicle_ids=",".join(ids), metric="dod_pct")
        assert set(result["series"].keys()) == set(ids)


# ----------------------------------------------------------------------
# get_vehicle_metadata -- asset registry lookup
# ----------------------------------------------------------------------
class TestVehicleMetadata:
    def test_known_vehicle_returns_asset_registry_fields(self, profile_df, tools):
        vid = profile_df["vehicle_id"].iloc[0]
        result = tools["get_vehicle_metadata"].fn(vehicle_id=vid)
        assert result["available"] is True
        assert result["vehicle_id"] == vid
        for field in ("make", "model", "vin", "purchase_date", "warranty_expiry_date"):
            assert result[field]

    def test_unknown_vehicle_reports_unavailable_rather_than_fabricating(self, tools):
        result = tools["get_vehicle_metadata"].fn(vehicle_id="EVR-9999")
        assert result["available"] is False
        assert "message" in result


# ----------------------------------------------------------------------
# Tool catalogue shape -- every tool exposes a name/description/params
# suitable for embedding in the system prompt (agent/tool_chat_engine.py)
# ----------------------------------------------------------------------
def test_every_tool_has_a_description(tools):
    for tool in tools.values():
        assert tool.description
        assert isinstance(tool.parameters, dict)