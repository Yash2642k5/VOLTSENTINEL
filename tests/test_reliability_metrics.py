"""
tests/test_reliability_metrics.py

Validates models/reliability_metrics.py's MTBF/MTTR computation against
both a hand-crafted DB (exact math) and a simulator-seeded fleet
(shape/no-crash, since seed_db.py never populates agent_actions).

Run from the project root:
    pytest tests/test_reliability_metrics.py -v
"""

import os

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator

from ingestion.db import get_connection, init_db, insert_telemetry_batch, insert_maintenance_batch
from ingestion.schemas import TelemetryReading, MaintenanceTicket
from agent.actions import init_actions_table

from models.reliability_metrics import ReliabilityAnalyzer

TEST_DB_PATH = os.path.join("data", "test_reliability_metrics.db")


def _insert_trigger(conn, vehicle_id, action_id, created_at):
    conn.execute(
        """INSERT INTO agent_actions
            (action_id, vehicle_id, action_type, priority, rationale, parameters, status, created_at)
            VALUES (?, ?, 'maintenance_trigger', 'medium', 'test', '{}', 'open', ?)""",
        (action_id, vehicle_id, created_at),
    )
    conn.commit()


def _insert_telemetry_row(conn, vehicle_id, cycle, timestamp):
    conn.execute(
        """INSERT INTO telemetry
            (vehicle_id, cycle, timestamp, capacity_kwh, capacity_pct_of_rated,
            rated_capacity_kwh, voltage, temperature_c, soc_pct, is_fast_charge, dod_pct)
            VALUES (?, ?, ?, 3.4, 97.0, 3.5, 51.0, 30.0, 80.0, 0, 60.0)""",
        (vehicle_id, cycle, timestamp),
    )
    conn.commit()


@pytest.fixture
def conn():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)
    init_actions_table(connection)
    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


# ----------------------------------------------------------------------
# analyze_vehicle — hand-crafted timestamps, exact math
# ----------------------------------------------------------------------
class TestAnalyzeVehicle:
    def test_no_history_is_insufficient_data(self, conn):
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.status == "insufficient_data"
        assert result.mtbf_hours is None
        assert result.mttr_hours is None
        assert result.maintenance_trigger_count == 0

    def test_single_trigger_has_no_mtbf(self, conn):
        _insert_trigger(conn, "EVR-0001", "ACT-1", "2026-01-01T00:00:00+00:00")
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.maintenance_trigger_count == 1
        assert result.mtbf_hours is None

    def test_mtbf_is_mean_gap_between_triggers(self, conn):
        _insert_trigger(conn, "EVR-0001", "ACT-1", "2026-01-01T00:00:00+00:00")
        _insert_trigger(conn, "EVR-0001", "ACT-2", "2026-01-02T00:00:00+00:00")  # +24h
        _insert_trigger(conn, "EVR-0001", "ACT-3", "2026-01-04T00:00:00+00:00")  # +48h
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.mtbf_hours == pytest.approx(36.0)

    def test_mttr_is_telemetry_gap_straddling_trigger(self, conn):
        _insert_telemetry_row(conn, "EVR-0001", 1, "2026-01-01T00:00:00")
        _insert_telemetry_row(conn, "EVR-0001", 2, "2026-01-01T10:00:00")  # last reading before fault
        _insert_trigger(conn, "EVR-0001", "ACT-1", "2026-01-01T12:00:00+00:00")
        _insert_telemetry_row(conn, "EVR-0001", 3, "2026-01-01T16:00:00")  # first reading after repair
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.mttr_hours == pytest.approx(6.0)  # 10:00 -> 16:00
        assert result.status == "ok"

    def test_trigger_with_no_telemetry_bracket_has_no_mttr(self, conn):
        _insert_telemetry_row(conn, "EVR-0001", 1, "2026-01-01T00:00:00")
        _insert_trigger(conn, "EVR-0001", "ACT-1", "2025-01-01T00:00:00+00:00")  # before all telemetry
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.mttr_hours is None

    def test_ticket_count_independent_of_triggers(self, conn):
        conn.execute(
            "INSERT INTO maintenance_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("TCK-1", "EVR-0001", "2026-01-01T00:00:00", 12.97, 77.59, "Scheduled inspection", "TECH-1"),
        )
        conn.commit()
        result = ReliabilityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.ticket_count == 1


# ----------------------------------------------------------------------
# analyze_fleet — simulator-seeded, no agent_actions history
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=5, num_cycles=20, random_seed=11)


@pytest.fixture(scope="module")
def seeded_conn(small_config):
    tgen = TelemetryGenerator(small_config)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(small_config)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    db_path = os.path.join("data", "test_reliability_metrics_fleet.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    connection = get_connection(db_path)
    init_db(connection)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**r) for r in tickets_df.to_dict(orient="records")]
    insert_telemetry_batch(connection, readings)
    insert_maintenance_batch(connection, tickets)

    yield connection
    connection.close()
    if os.path.exists(db_path):
        os.remove(db_path)


class TestAnalyzeFleet:
    def test_one_row_per_vehicle(self, small_config, seeded_conn):
        df = ReliabilityAnalyzer().analyze_fleet(seeded_conn)
        assert len(df) == small_config.fleet_size

    def test_no_agent_history_means_insufficient_data(self, seeded_conn):
        df = ReliabilityAnalyzer().analyze_fleet(seeded_conn)
        assert (df["status"] == "insufficient_data").all()

    def test_ticket_counts_reflect_seeded_tickets(self, seeded_conn):
        df = ReliabilityAnalyzer().analyze_fleet(seeded_conn)
        assert (df["ticket_count"] > 0).all()

    def test_empty_db_returns_empty_frame(self, conn):
        df = ReliabilityAnalyzer().analyze_fleet(conn)
        assert df.empty
