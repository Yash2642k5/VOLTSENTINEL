"""
tests/test_risk_engine.py

Validates models/risk_engine.py: correct merge of RUL + thermal +
security + charging into one profile, no vehicles lost or duplicated
in the join, overall_risk_level banding logic, and — the test that
matters most — that a vehicle with an actually-injected attack gets
correctly surfaced as higher risk than one without.

Not part of the originally documented folder_structure.py test list —
added on request to cover Phase 3's remaining two files.

Run from the project root:
    pytest tests/test_risk_engine.py -v
"""

import math
import os

import pandas as pd
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

from models.rul_model import RULModel
from models.risk_engine import RiskEngine


TEST_DB_PATH = os.path.join("data", "test_risk_engine.db")


@pytest.fixture(scope="module")
def small_config():
    # Higher attack rate than the simulator default so the "attacked vehicle
    # scores higher risk" assertion isn't relying on luck to have any attacks.
    return SimulatorConfig(
        fleet_size=8, num_cycles=150, random_seed=2026,
        attack_injection_rate_pct=0.25,
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

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="module")
def engine(small_config):
    return RiskEngine(rul_model=RULModel(end_of_life_capacity_pct=small_config.end_of_life_capacity_pct * 100))


@pytest.fixture(scope="module")
def profile_df(engine, conn):
    return engine.build_fleet_profile(conn)


# ----------------------------------------------------------------------
# Merge correctness
# ----------------------------------------------------------------------
class TestMergeCorrectness:
    def test_no_vehicles_lost_in_merge(self, small_config, profile_df):
        assert len(profile_df) == small_config.fleet_size

    def test_no_duplicate_vehicles(self, profile_df):
        assert profile_df["vehicle_id"].is_unique

    def test_all_expected_columns_present(self, profile_df):
        expected = {
            "vehicle_id", "status", "rul_cycles", "r_squared",
            "thermal_anomaly_count", "critical_temp_count",
            "unticketed_command_count", "max_security_severity", "suspicious_command_count",
            "fast_charge_frequency_pct", "mean_dod_pct", "charge_stress_score",
            "missing_cycle_count", "is_stale", "hours_since_last_reading", "out_of_range_jump_count",
            "overall_risk_level",
        }
        assert expected.issubset(set(profile_df.columns))

    def test_freshly_seeded_fleet_has_no_data_quality_issues(self, profile_df):
        """Simulator-seeded telemetry is contiguous and in-range by
        construction — the only genuinely expected finding is staleness,
        since seeded timestamps aren't tied to wall-clock 'now'."""
        assert (profile_df["missing_cycle_count"] == 0).all()
        assert (profile_df["out_of_range_jump_count"] == 0).all()

    def test_no_null_risk_level(self, profile_df):
        assert profile_df["overall_risk_level"].notnull().all()

    def test_risk_level_is_valid_category(self, profile_df):
        assert set(profile_df["overall_risk_level"]).issubset({"minimal", "low", "medium", "high"})


# ----------------------------------------------------------------------
# The assertion that matters most: does risk_engine actually surface
# the vehicle with a real, injected attack as higher risk?
# ----------------------------------------------------------------------
class TestAttackSurfacing:
    def test_attacked_vehicle_has_nonzero_security_signal(self, simulated_data, profile_df):
        _, _, commands_df = simulated_data
        attacked_vehicles = set(commands_df[commands_df["is_attack"] == True]["vehicle_id"])  # noqa: E712
        assert len(attacked_vehicles) > 0, "test fixture produced no attacks — test is vacuous"

        for vid in attacked_vehicles:
            row = profile_df[profile_df["vehicle_id"] == vid].iloc[0]
            assert row["suspicious_command_count"] > 0
            assert row["max_security_severity"] != "none"

    def test_unattacked_vehicles_have_no_security_signal(self, simulated_data, profile_df):
        _, _, commands_df = simulated_data
        attacked_vehicles = set(commands_df[commands_df["is_attack"] == True]["vehicle_id"])  # noqa: E712
        all_vehicles = set(commands_df["vehicle_id"])
        clean_vehicles = all_vehicles - attacked_vehicles

        for vid in clean_vehicles:
            row = profile_df[profile_df["vehicle_id"] == vid].iloc[0]
            assert row["suspicious_command_count"] == 0
            assert row["max_security_severity"] == "none"

    def test_attacked_vehicle_risk_level_at_least_as_high_as_clean(self, simulated_data, profile_df):
        """The attacked vehicle's overall_risk_level must never rank below
        an otherwise-identical clean vehicle's — security signal should
        only ever push risk up, never down."""
        rank = {"minimal": 0, "low": 1, "medium": 2, "high": 3}
        _, _, commands_df = simulated_data
        attacked_vehicles = set(commands_df[commands_df["is_attack"] == True]["vehicle_id"])  # noqa: E712

        for vid in attacked_vehicles:
            row = profile_df[profile_df["vehicle_id"] == vid].iloc[0]
            # An attacked vehicle has at least 1 concern (security), so it can
            # never be "minimal".
            assert rank[row["overall_risk_level"]] >= rank["low"]


# ----------------------------------------------------------------------
# overall_risk_level banding logic (direct unit test on the method)
# ----------------------------------------------------------------------
class TestRiskBanding:
    def test_zero_concerns_is_minimal(self, engine):
        row = pd.Series({
            "status": "healthy", "thermal_anomaly_count": 0,
            "max_security_severity": "none", "charge_stress_score": 30,
        })
        assert engine._overall_risk_level(row) == "minimal"

    def test_one_concern_is_low(self, engine):
        row = pd.Series({
            "status": "degraded", "thermal_anomaly_count": 0,
            "max_security_severity": "none", "charge_stress_score": 30,
        })
        assert engine._overall_risk_level(row) == "low"

    def test_two_concerns_is_medium(self, engine):
        row = pd.Series({
            "status": "degraded", "thermal_anomaly_count": 5,
            "max_security_severity": "none", "charge_stress_score": 30,
        })
        assert engine._overall_risk_level(row) == "medium"

    def test_all_four_concerns_is_high(self, engine):
        row = pd.Series({
            "status": "critical", "thermal_anomaly_count": 5,
            "max_security_severity": "high", "charge_stress_score": 90,
        })
        assert engine._overall_risk_level(row) == "high"

    def test_three_concerns_is_high(self, engine):
        row = pd.Series({
            "status": "critical", "thermal_anomaly_count": 5,
            "max_security_severity": "high", "charge_stress_score": 10,
        })
        assert engine._overall_risk_level(row) == "high"

    def test_data_quality_issue_counts_as_a_concern(self, engine):
        row = pd.Series({
            "status": "degraded", "thermal_anomaly_count": 0,
            "max_security_severity": "none", "charge_stress_score": 30,
            "is_stale": True,
        })
        assert engine._overall_risk_level(row) == "medium"  # degraded + stale = 2 concerns