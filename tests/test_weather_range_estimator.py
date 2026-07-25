"""
tests/test_weather_range_estimator.py

Validates the Feature 8 addition to models/range_estimator.py: the
weather-adjustment curve itself, and that estimate_vehicle()'s existing
callers (no ambient_temp_c passed) are completely unaffected — matching
the existing style of tests/test_range_estimator.py, which this file
deliberately does not modify.

No real network calls — a scripted FakeWeatherClient plays the
weather API's part, same pattern as tests/test_decision_engine.py's
FakeGeminiClient.

Run from the project root:
    pytest tests/test_weather_range_estimator.py -v
"""

import os

import pytest

from simulator.config import SimulatorConfig

from models.range_estimator import RangeEstimator, DEFAULT_DEPOT_LOCATIONS


def _row(cycle, soc_pct, capacity_kwh=3.0):
    return {"cycle": cycle, "soc_pct": soc_pct, "capacity_kwh": capacity_kwh}


class FakeWeatherClient:
    """Returns a fixed temperature regardless of coordinate — enough to
    exercise the adjustment math without a real HTTP call."""

    def __init__(self, temp_c):
        self.temp_c = temp_c
        self.calls = []

    def get_current_temperature_c(self, latitude, longitude):
        self.calls.append((latitude, longitude))
        return self.temp_c


@pytest.fixture
def estimator():
    return RangeEstimator(kwh_per_km=0.06, low_range_threshold_km=15.0, low_soc_threshold_pct=20.0)


# ----------------------------------------------------------------------
# Backward compatibility — no ambient_temp_c passed at all
# ----------------------------------------------------------------------
class TestNoWeatherIsUnchanged:
    def test_omitting_ambient_temp_gives_factor_one(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.weather_adjustment_factor == pytest.approx(1.0)
        assert result.ambient_temp_c is None
        assert result.estimated_range_km == pytest.approx(25.0)  # same number as before Feature 8

    def test_explicit_none_ambient_temp_is_also_factor_one(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, ambient_temp_c=None)
        assert result.weather_adjustment_factor == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Adjustment curve
# ----------------------------------------------------------------------
class TestWeatherAdjustmentCurve:
    def test_mild_temperature_no_penalty(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, ambient_temp_c=22.0)
        assert result.weather_adjustment_factor == pytest.approx(1.0)

    def test_cold_temperature_increases_kwh_per_km(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, ambient_temp_c=5.0)
        # 10°C below the 15°C cool reference * 1.5%/°C = +15%
        assert result.weather_adjustment_factor == pytest.approx(1.15)
        assert result.kwh_per_km == pytest.approx(0.06 * 1.15)

    def test_hot_temperature_increases_kwh_per_km(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, ambient_temp_c=40.0)
        # 10°C above the 30°C warm reference * 0.8%/°C = +8%
        assert result.weather_adjustment_factor == pytest.approx(1.08)

    def test_cold_range_is_shorter_than_mild_range(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        cold = estimator.estimate_vehicle("V1", rows, ambient_temp_c=0.0)
        mild = estimator.estimate_vehicle("V1", rows, ambient_temp_c=22.0)
        assert cold.estimated_range_km < mild.estimated_range_km

    def test_extreme_temperature_is_capped(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, ambient_temp_c=-40.0)
        assert result.weather_adjustment_factor <= estimator.max_weather_adjustment_factor

    def test_empty_telemetry_still_reports_weather_fields(self, estimator):
        result = estimator.estimate_vehicle("V-EMPTY", [], ambient_temp_c=10.0)
        assert result.estimated_range_km is None
        assert result.ambient_temp_c == 10.0
        assert result.weather_adjustment_factor > 1.0


# ----------------------------------------------------------------------
# Live wrappers (estimate_vehicle_live / estimate_fleet) — DB + fake
# weather client, no real network
# ----------------------------------------------------------------------
TEST_DB = os.path.join("data", "test_weather_range_estimator.db")


@pytest.fixture
def conn():
    """Fresh SQLite DB per test, seeded with telemetry for a small fleet
    plus one legitimate (ticketed) command for EVR-0001 near the
    Bengaluru depot, so estimate_vehicle_live() has a real location to
    map to a depot.

    The command MUST reference a real maintenance_tickets row first —
    commands.ticket_id is a genuine foreign key (ingestion/db.py), so a
    command pointing at a ticket_id that was never inserted raises
    sqlite3.IntegrityError, exactly as it should for a real client too.

    Uses yield + a finally-style teardown (matching every other DB
    fixture in this test suite, e.g. tests/test_risk_engine.py) so the
    connection is ALWAYS closed and the file ALWAYS removed even if the
    test itself fails — on Windows, sqlite3 holds an OS-level file lock
    while the connection is open, so a prior test's unclosed connection
    causes every subsequent test's os.remove() to raise PermissionError,
    which is exactly what happened before this fixture existed."""
    from ingestion.db import (
        get_connection, init_db, insert_telemetry_batch, insert_maintenance_batch,
        insert_command_batch,
    )
    from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent
    from simulator.telemetry_generator import TelemetryGenerator

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    connection = get_connection(TEST_DB)
    init_db(connection)

    cfg = SimulatorConfig(fleet_size=3, num_cycles=20, random_seed=5)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    insert_telemetry_batch(connection, readings)

    depot_lat, depot_lon = DEFAULT_DEPOT_LOCATIONS[0]
    ticket = MaintenanceTicket(
        ticket_id="TCK-TEST01", vehicle_id="EVR-0001", timestamp="2026-01-01T00:00:00",
        depot_lat=depot_lat, depot_lon=depot_lon, reason="Scheduled inspection",
        technician_id="TECH-100",
    )
    insert_maintenance_batch(connection, [ticket])

    command = CommandEvent(
        command_id="CMD-TESTFIX01", vehicle_id="EVR-0001",
        timestamp="2026-01-01T00:30:00", command_type="enable",
        latitude=depot_lat, longitude=depot_lon, ticket_id="TCK-TEST01",
    )
    insert_command_batch(connection, [command])

    yield connection

    connection.close()
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestLiveWrappers:
    def test_estimate_vehicle_live_uses_weather_client(self, estimator, conn):
        fake_weather = FakeWeatherClient(temp_c=40.0)

        result = estimator.estimate_vehicle_live(conn, "EVR-0001", weather_client=fake_weather)

        assert result.ambient_temp_c == 40.0
        assert result.weather_adjustment_factor > 1.0
        assert len(fake_weather.calls) == 1

    def test_no_weather_client_gives_unchanged_behaviour(self, estimator, conn):
        result = estimator.estimate_vehicle_live(conn, "EVR-0001", weather_client=None)

        assert result.ambient_temp_c is None
        assert result.weather_adjustment_factor == pytest.approx(1.0)

    def test_vehicle_with_no_command_history_falls_back_gracefully(self, estimator, conn):
        fake_weather = FakeWeatherClient(temp_c=40.0)

        # EVR-0002 has telemetry but no command history in this fixture.
        result = estimator.estimate_vehicle_live(conn, "EVR-0002", weather_client=fake_weather)

        assert result.ambient_temp_c is None
        assert result.weather_adjustment_factor == pytest.approx(1.0)
        assert fake_weather.calls == []

    def test_estimate_fleet_with_weather_client_returns_expected_columns(self, estimator, conn):
        fake_weather = FakeWeatherClient(temp_c=35.0)

        df = estimator.estimate_fleet(conn, weather_client=fake_weather)

        assert "ambient_temp_c" in df.columns
        assert "weather_adjustment_factor" in df.columns
        assert len(df) == 3  # fleet_size from the conn fixture's SimulatorConfig