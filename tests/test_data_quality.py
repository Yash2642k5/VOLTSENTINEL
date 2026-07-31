"""
tests/test_data_quality.py

Validates models/data_quality.py's missing-cycle, stale-sensor, and
out-of-range checks against a hand-crafted DB (exact control over
gaps/timestamps/values) and a simulator-seeded fleet (shape/no-crash).

Run from the project root:
    pytest tests/test_data_quality.py -v
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from ingestion.db import get_connection, init_db
from models.data_quality import DataQualityAnalyzer

TEST_DB_PATH = os.path.join("data", "test_data_quality.db")


def _insert_telemetry(conn, vehicle_id, cycle, timestamp, **overrides):
    row = {
        "capacity_kwh": 3.4, "capacity_pct_of_rated": 97.0, "rated_capacity_kwh": 3.5,
        "voltage": 51.0, "temperature_c": 30.0, "soc_pct": 80.0,
        "is_fast_charge": 0, "dod_pct": 60.0,
    }
    row.update(overrides)
    conn.execute(
        """INSERT INTO telemetry
            (vehicle_id, cycle, timestamp, capacity_kwh, capacity_pct_of_rated,
            rated_capacity_kwh, voltage, temperature_c, soc_pct, is_fast_charge, dod_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            vehicle_id, cycle, timestamp, row["capacity_kwh"], row["capacity_pct_of_rated"],
            row["rated_capacity_kwh"], row["voltage"], row["temperature_c"], row["soc_pct"],
            row["is_fast_charge"], row["dod_pct"],
        ),
    )
    conn.commit()


@pytest.fixture
def conn():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)
    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


class TestAnalyzeVehicle:
    def test_no_telemetry_is_flagged_stale_with_no_telemetry_issue(self, conn):
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001")
        assert result.is_stale is True
        assert "no_telemetry" in result.issues
        assert result.missing_cycle_count == 0

    def test_contiguous_recent_cycles_have_no_issues(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        for cycle in range(1, 6):
            ts = (now - timedelta(hours=(6 - cycle) * 4)).isoformat()
            _insert_telemetry(conn, "EVR-0001", cycle, ts)
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.missing_cycle_count == 0
        assert result.is_stale is False
        assert result.out_of_range_jump_count == 0
        assert result.issues == []

    def test_gap_in_cycle_sequence_is_detected(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        for cycle in (1, 2, 5, 6):  # missing 3, 4
            _insert_telemetry(conn, "EVR-0001", cycle, now.isoformat())
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.missing_cycle_count == 2
        assert "missing_cycles" in result.issues

    def test_old_last_reading_is_stale(self, conn):
        now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, (now - timedelta(hours=100)).isoformat())
        result = DataQualityAnalyzer(stale_hours=48.0).analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.is_stale is True
        assert result.hours_since_last_reading == pytest.approx(100.0, abs=0.2)
        assert "stale_sensor" in result.issues

    def test_recent_last_reading_is_not_stale(self, conn):
        now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, (now - timedelta(hours=2)).isoformat())
        result = DataQualityAnalyzer(stale_hours=48.0).analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.is_stale is False

    def test_capacity_increase_jump_is_flagged(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, now.isoformat(), capacity_pct_of_rated=90.0)
        _insert_telemetry(conn, "EVR-0001", 2, now.isoformat(), capacity_pct_of_rated=95.0)  # +5pp jump
        result = DataQualityAnalyzer(capacity_increase_jump_pct=2.0).analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.out_of_range_jump_count == 1
        assert "out_of_range_values" in result.issues

    def test_out_of_bounds_temperature_is_flagged(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, now.isoformat(), temperature_c=999.0)
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.out_of_range_jump_count == 1

    def test_out_of_bounds_voltage_is_flagged(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, now.isoformat(), voltage=-5.0)
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.out_of_range_jump_count == 1

    def test_out_of_bounds_soc_is_flagged(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, now.isoformat(), soc_pct=150.0)
        result = DataQualityAnalyzer().analyze_vehicle(conn, "EVR-0001", now=now)
        assert result.out_of_range_jump_count == 1


class TestAnalyzeFleet:
    def test_empty_db_returns_empty_frame(self, conn):
        df = DataQualityAnalyzer().analyze_fleet(conn)
        assert df.empty

    def test_one_row_per_vehicle(self, conn):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        _insert_telemetry(conn, "EVR-0001", 1, now.isoformat())
        _insert_telemetry(conn, "EVR-0002", 1, now.isoformat())
        df = DataQualityAnalyzer().analyze_fleet(conn, now=now)
        assert set(df["vehicle_id"]) == {"EVR-0001", "EVR-0002"}
