"""
tests/test_live_feed.py

Validates simulator/live_feed.py: resuming from an existing seeded
fleet's latest cycle, inserting plausible continuation readings, and
running a bounded number of ticks without blocking.

Run from the project root:
    pytest tests/test_live_feed.py -v
"""

import math
import os
from datetime import datetime, timedelta, timezone

import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.live_feed import LiveTelemetryFeed

from ingestion.db import (
    get_connection, init_db, insert_telemetry_batch, get_telemetry_for_vehicle,
    upsert_vehicle_live_state, get_all_vehicle_live_state,
)
from ingestion.schemas import TelemetryReading

TEST_DB_PATH = os.path.join("data", "test_live_feed.db")


@pytest.fixture
def small_config():
    return SimulatorConfig(fleet_size=3, num_cycles=50, random_seed=13)


@pytest.fixture
def empty_conn():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)
    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def seeded_conn(empty_conn, small_config):
    tgen = TelemetryGenerator(small_config)
    telem_df = tgen.generate_fleet()
    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    insert_telemetry_batch(empty_conn, readings)
    return empty_conn


class TestPrimeFromDb:
    def test_empty_db_starts_every_vehicle_at_cycle_one(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        assert all(c == 1 for c in feed._next_cycle.values())
        assert all(s == 0.0 for s in feed._stress_cycles.values())

    def test_seeded_db_resumes_after_last_cycle(self, small_config, seeded_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(seeded_conn)
        for vid in feed.tgen.vehicle_ids:
            last_row = get_telemetry_for_vehicle(seeded_conn, vid)[-1]
            assert feed._next_cycle[vid] == last_row["cycle"] + 1

    def test_decayed_vehicle_has_positive_stress_cycles(self, small_config, seeded_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(seeded_conn)
        assert all(s > 0 for s in feed._stress_cycles.values())


class TestTick:
    def test_inserts_one_row_per_vehicle(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        inserted = feed.tick(empty_conn)
        assert len(inserted) == small_config.fleet_size

    def test_cycle_increments_and_no_duplicates(self, small_config, seeded_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(seeded_conn)
        vid = feed.tgen.vehicle_ids[0]
        before_count = len(get_telemetry_for_vehicle(seeded_conn, vid))

        feed.tick(seeded_conn)
        feed.tick(seeded_conn)

        rows = get_telemetry_for_vehicle(seeded_conn, vid)
        assert len(rows) == before_count + 2
        cycles = [r["cycle"] for r in rows]
        assert len(cycles) == len(set(cycles))  # no duplicate cycle numbers

    def test_new_reading_continues_plausibly_from_last_known_capacity(self, small_config, seeded_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(seeded_conn)
        vid = feed.tgen.vehicle_ids[0]
        last_capacity_pct = get_telemetry_for_vehicle(seeded_conn, vid)[-1]["capacity_pct_of_rated"]

        feed.tick(seeded_conn)

        new_capacity_pct = get_telemetry_for_vehicle(seeded_conn, vid)[-1]["capacity_pct_of_rated"]
        assert abs(new_capacity_pct - last_capacity_pct) < 10.0  # one cycle shouldn't jump wildly


class TestRunForever:
    def test_stops_after_max_ticks(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.run_forever(empty_conn, interval_seconds=0, max_ticks=3)
        vid = feed.tgen.vehicle_ids[0]
        assert len(get_telemetry_for_vehicle(empty_conn, vid)) == 3


# ----------------------------------------------------------------------
# Movement / activity status
# ----------------------------------------------------------------------
class TestPrimePosition:
    def test_fresh_vehicle_starts_near_its_home_depot(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        for vid in feed.tgen.vehicle_ids:
            home = feed._home[vid]
            lat, lon = feed._position[vid]
            assert math.hypot(lat - home[0], lon - home[1]) < 0.05
            assert feed._status[vid] == "active"

    def test_resumes_position_from_existing_live_state(self, small_config, empty_conn):
        vid = TelemetryGenerator(small_config).vehicle_ids[0]
        now = datetime.now(timezone.utc)
        upsert_vehicle_live_state(
            empty_conn, vid, 12.34, 56.78, "inactive",
            (now - timedelta(minutes=10)).isoformat(), now.isoformat(),
        )
        empty_conn.commit()

        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        assert feed._position[vid] == (12.34, 56.78)
        assert feed._status[vid] == "inactive"


class TestAdvancePosition:
    def test_moving_vehicle_changes_position(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config, move_probability=1.0)
        feed.prime_from_db(empty_conn)
        vid = feed.tgen.vehicle_ids[0]
        before = feed._position[vid]
        feed._advance_position(vid, datetime.now(timezone.utc))
        assert feed._position[vid] != before
        assert feed._status[vid] == "active"

    def test_far_from_home_pulls_back_toward_depot(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config, move_probability=1.0, step_deg=0.0)
        feed.prime_from_db(empty_conn)
        vid = feed.tgen.vehicle_ids[0]
        home = feed._home[vid]
        feed._position[vid] = (home[0] + 1.0, home[1] + 1.0)  # far outside operating_radius_deg

        dist_before = math.hypot(*(a - b for a, b in zip(feed._position[vid], home)))
        feed._advance_position(vid, datetime.now(timezone.utc))
        dist_after = math.hypot(*(a - b for a, b in zip(feed._position[vid], home)))
        assert dist_after < dist_before

    def test_idle_past_threshold_becomes_inactive(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config, move_probability=0.0, inactive_after_seconds=180.0)
        feed.prime_from_db(empty_conn)
        vid = feed.tgen.vehicle_ids[0]
        feed._last_moved_at[vid] = datetime.now(timezone.utc) - timedelta(minutes=5)

        feed._advance_position(vid, datetime.now(timezone.utc))
        assert feed._status[vid] == "inactive"

    def test_idle_within_threshold_stays_active(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config, move_probability=0.0, inactive_after_seconds=180.0)
        feed.prime_from_db(empty_conn)
        vid = feed.tgen.vehicle_ids[0]
        feed._last_moved_at[vid] = datetime.now(timezone.utc) - timedelta(minutes=1)

        feed._advance_position(vid, datetime.now(timezone.utc))
        assert feed._status[vid] == "active"


class TestTickLiveState:
    def test_writes_one_live_state_row_per_vehicle(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        feed.tick(empty_conn)
        assert len(get_all_vehicle_live_state(empty_conn)) == small_config.fleet_size

    def test_live_state_status_is_valid(self, small_config, empty_conn):
        feed = LiveTelemetryFeed(small_config)
        feed.prime_from_db(empty_conn)
        feed.tick(empty_conn)
        for row in get_all_vehicle_live_state(empty_conn):
            assert row["status"] in ("active", "inactive")
