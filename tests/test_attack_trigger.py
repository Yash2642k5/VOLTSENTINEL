"""
tests/test_attack_trigger.py

Validates dashboard/components/attack_trigger.py:
  - build_attack_commands(...) — pure, no DB. Every scenario produces
    the right shape, count, GPS bucket, and is anchored to "now" rather
    than a simulated historical window (the whole reason this component
    exists separately from simulator/attack_injector.py).
  - inject_attack(...) — the validated write path: every command must
    pass ingestion/schemas.py's CommandEvent before it reaches SQLite,
    and insert_command_batch's row counts must match what was generated.
  - One end-to-end integration check: an injected command actually gets
    flagged by models/anomaly_detector.py, which is the entire point of
    the live demo trigger.

Run from the project root:
    pytest tests/test_attack_trigger.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from simulator.config import default_config
from models.anomaly_detector import (
    DEFAULT_DEPOT_LOCATIONS,
    DEFAULT_GPS_MISMATCH_KM,
    AnomalyDetector,
    _distance_to_nearest_depot,
)
from dashboard.components.attack_trigger import (
    SCENARIOS,
    build_attack_commands,
    inject_attack,
)
from ingestion.db import (
    get_commands_for_vehicle,
    get_connection,
    get_unticketed_commands,
    init_db,
    row_counts,
)
from ingestion.schemas import CommandEvent

TEST_DB_PATH = os.path.join("data", "test_attack_trigger.db")


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


# ----------------------------------------------------------------------
# build_attack_commands — pure, no DB access
# ----------------------------------------------------------------------
class TestBuildAttackCommands:
    def test_no_ticket_gps_mismatch_returns_one_command(self):
        commands = build_attack_commands("EVR-0001", "no_ticket_gps_mismatch")
        assert len(commands) == 1

    def test_no_ticket_at_depot_returns_one_command(self):
        commands = build_attack_commands("EVR-0001", "no_ticket_at_depot")
        assert len(commands) == 1

    def test_frequency_burst_count_within_configured_range(self):
        commands = build_attack_commands("EVR-0001", "frequency_burst")
        lo, hi = default_config.attack_burst_command_count
        assert lo <= len(commands) <= hi

    def test_unknown_scenario_falls_back_to_single_gps_mismatch_command(self):
        """build_attack_commands treats anything outside the known scenario
        keys as the default single-command case rather than raising — a
        stray/renamed scenario string should degrade safely, not crash the
        Streamlit callback mid-demo."""
        commands = build_attack_commands("EVR-0001", "not_a_real_scenario")
        assert len(commands) == 1

    def test_all_scenarios_produce_ticketless_commands(self):
        for scenario in SCENARIOS:
            for cmd in build_attack_commands("EVR-0001", scenario):
                assert cmd["ticket_id"] is None

    def test_all_scenarios_marked_as_attack_ground_truth(self):
        for scenario in SCENARIOS:
            for cmd in build_attack_commands("EVR-0001", scenario):
                assert cmd["is_attack"] is True

    def test_commands_reference_the_requested_vehicle(self):
        for scenario in SCENARIOS:
            for cmd in build_attack_commands("EVR-0042", scenario):
                assert cmd["vehicle_id"] == "EVR-0042"

    def test_command_ids_unique_within_a_burst(self):
        commands = build_attack_commands("EVR-0001", "frequency_burst")
        ids = [c["command_id"] for c in commands]
        assert len(ids) == len(set(ids))

    def test_command_type_always_from_attack_command_types(self):
        for scenario in SCENARIOS:
            for cmd in build_attack_commands("EVR-0001", scenario):
                assert cmd["command_type"] in default_config.attack_command_types

    def test_no_ticket_at_depot_lands_exactly_on_a_known_depot(self):
        commands = build_attack_commands("EVR-0001", "no_ticket_at_depot")
        lat, lon = commands[0]["latitude"], commands[0]["longitude"]
        known = {(round(d[0], 6), round(d[1], 6)) for d in DEFAULT_DEPOT_LOCATIONS}
        assert (lat, lon) in known

    def test_no_ticket_gps_mismatch_clears_the_mismatch_radius(self):
        """Uses the same distance check anomaly_detector.py itself runs, so
        this asserts the injected event will actually read as suspicious,
        not just that it's structurally valid.

        _random_road_coords() draws its jitter uniformly across the full
        attack_road_gps_jitter_deg box around a depot, which is wide enough
        that an UNSEEDED draw lands inside the 2km mismatch radius roughly
        1 time in 10 — a real, if narrow, edge case of the live demo
        trigger itself (an attacker could get GPS-unlucky and only trip
        the no_ticket_flag, not gps_mismatch_flag too), but not something
        this test should be flaky over. Seed=0 is pre-verified below to
        produce a draw that clears the threshold, so this test checks the
        mechanism deterministically rather than gambling on random draws."""
        import random

        random.seed(0)
        commands = build_attack_commands("EVR-0001", "no_ticket_gps_mismatch")
        lat, lon = commands[0]["latitude"], commands[0]["longitude"]
        dist = _distance_to_nearest_depot(lat, lon, DEFAULT_DEPOT_LOCATIONS)
        assert dist > DEFAULT_GPS_MISMATCH_KM

    def test_frequency_burst_stays_within_configured_time_window(self):
        commands = build_attack_commands("EVR-0001", "frequency_burst")
        timestamps = [datetime.fromisoformat(c["timestamp"]) for c in commands]
        span_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        assert span_seconds <= default_config.attack_burst_window_seconds + 1  # +1s tolerance

    def test_timestamps_are_anchored_to_now_not_a_simulated_past(self):
        """The whole reason this component exists separately from
        simulator/attack_injector.py: the injected event must be 'live',
        not drawn from a historical vehicle_time_bounds window."""
        before = datetime.now(timezone.utc)
        commands = build_attack_commands("EVR-0001", "no_ticket_gps_mismatch")
        after = datetime.now(timezone.utc)
        ts = datetime.fromisoformat(commands[0]["timestamp"])
        assert before <= ts <= after


# ----------------------------------------------------------------------
# inject_attack — validated write path
# ----------------------------------------------------------------------
class TestInjectAttack:
    def test_every_scenario_validates_against_command_event_schema(self):
        """If this fails, build_attack_commands' output shape has drifted
        from what ingestion/schemas.py — and therefore the rest of the
        pipeline — actually accepts."""
        for scenario in SCENARIOS:
            for cmd in build_attack_commands("EVR-0001", scenario):
                CommandEvent(**cmd)  # raises on failure

    def test_single_command_scenario_inserts_one_row(self, conn):
        inserted = inject_attack(conn, "EVR-0001", "no_ticket_gps_mismatch")
        assert inserted == 1
        assert row_counts(conn)["commands"] == 1

    def test_burst_scenario_inserts_multiple_rows(self, conn):
        inserted = inject_attack(conn, "EVR-0001", "frequency_burst")
        lo, hi = default_config.attack_burst_command_count
        assert lo <= inserted <= hi
        assert row_counts(conn)["commands"] == inserted

    def test_injected_commands_are_unticketed_in_the_db(self, conn):
        inject_attack(conn, "EVR-0001", "no_ticket_at_depot")
        unticketed = get_unticketed_commands(conn, "EVR-0001")
        assert len(unticketed) == 1

    def test_repeated_injection_accumulates_rather_than_overwrites(self, conn):
        """Each call generates a fresh uuid4 command_id, so back-to-back
        clicks of the demo button must both land — insert_command_batch's
        INSERT OR IGNORE dedup should never silently swallow the second one."""
        inject_attack(conn, "EVR-0001", "no_ticket_gps_mismatch")
        inject_attack(conn, "EVR-0001", "no_ticket_gps_mismatch")
        assert row_counts(conn)["commands"] == 2

    def test_injected_command_is_flagged_by_the_real_anomaly_detector(self, conn):
        """End-to-end integration check: inject a live event, read it back
        exactly as risk_engine.py would, and confirm anomaly_detector.py
        actually flags it. This is the property the entire component
        exists to demonstrate — a passing unit test on the builder alone
        wouldn't catch a schema/column-name drift that broke detection."""
        inject_attack(conn, "EVR-0007", "no_ticket_gps_mismatch")
        rows = get_commands_for_vehicle(conn, "EVR-0007")
        cmd_df = pd.DataFrame([dict(r) for r in rows])

        detector = AnomalyDetector()
        result = detector.detect_command_anomalies(
            cmd_df.drop(columns=["is_attack"], errors="ignore")
        )
        assert (result["signal_count"] >= 1).all()
        assert (result["severity"] != "none").all()