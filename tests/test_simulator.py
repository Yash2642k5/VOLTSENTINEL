"""
tests/test_simulator.py

Validates Phase 1 (simulator/) output shape and behaviour before anything
downstream (ingestion, models, agent) is built on top of it.

Run from the project root:
    pytest tests/test_simulator.py -v
"""

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector


# Small, fast config for tests — no need to run the full 50-vehicle / 500-cycle default.
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=8, num_cycles=60, random_seed=7)


@pytest.fixture(scope="module")
def telemetry(small_config):
    gen = TelemetryGenerator(small_config)
    return gen, gen.generate_fleet()


@pytest.fixture(scope="module")
def tickets(small_config, telemetry):
    gen, fleet_df = telemetry
    bounds = gen.get_vehicle_time_bounds(fleet_df)
    mgen = MaintenanceGenerator(small_config)
    return mgen.generate_fleet_tickets(bounds)


@pytest.fixture(scope="module")
def commands(small_config, telemetry, tickets):
    gen, fleet_df = telemetry
    bounds = gen.get_vehicle_time_bounds(fleet_df)
    ainj = AttackInjector(small_config)
    return ainj.generate_command_stream(bounds, tickets)


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
class TestTelemetryGenerator:
    def test_row_count(self, small_config, telemetry):
        _, df = telemetry
        assert len(df) == small_config.fleet_size * small_config.num_cycles

    def test_vehicle_count(self, small_config, telemetry):
        _, df = telemetry
        assert df["vehicle_id"].nunique() == small_config.fleet_size

    def test_no_nulls(self, telemetry):
        _, df = telemetry
        assert not df.isnull().values.any()

    def test_capacity_trends_downward(self, telemetry):
        """Capacity should generally fade over cycles, not trend up."""
        _, df = telemetry
        vid = df["vehicle_id"].iloc[0]
        vdf = df[df.vehicle_id == vid].sort_values("cycle")
        first_10_avg = vdf.head(10)["capacity_pct_of_rated"].mean()
        last_10_avg = vdf.tail(10)["capacity_pct_of_rated"].mean()
        assert last_10_avg < first_10_avg

    def test_capacity_stays_positive(self, telemetry):
        _, df = telemetry
        assert (df["capacity_kwh"] > 0).all()

    def test_soc_within_bounds(self, small_config, telemetry):
        _, df = telemetry
        assert df["soc_pct"].between(
            small_config.soc_min_pct - 1, small_config.soc_max_pct + 1
        ).all()

    def test_dod_within_plausible_range(self, telemetry):
        _, df = telemetry
        assert df["dod_pct"].between(0, 100).all()

    def test_timestamps_increase_per_vehicle(self, telemetry):
        _, df = telemetry
        vid = df["vehicle_id"].iloc[0]
        ts = pd.to_datetime(df[df.vehicle_id == vid].sort_values("cycle")["timestamp"])
        assert ts.is_monotonic_increasing

    def test_get_vehicle_time_bounds(self, small_config, telemetry):
        gen, df = telemetry
        bounds = gen.get_vehicle_time_bounds(df)
        assert len(bounds) == small_config.fleet_size
        for start, end in bounds.values():
            assert start <= end


# ----------------------------------------------------------------------
# Maintenance tickets
# ----------------------------------------------------------------------
class TestMaintenanceGenerator:
    def test_tickets_generated(self, tickets):
        assert len(tickets) > 0

    def test_ticket_ids_unique(self, tickets):
        assert tickets["ticket_id"].is_unique

    def test_tickets_reference_valid_vehicles(self, small_config, tickets):
        gen = TelemetryGenerator(small_config)
        assert set(tickets["vehicle_id"]).issubset(set(gen.vehicle_ids))

    def test_ticket_timestamps_within_vehicle_bounds(self, small_config, telemetry, tickets):
        gen, df = telemetry
        bounds = gen.get_vehicle_time_bounds(df)
        ts = pd.to_datetime(tickets["timestamp"])
        for i, row in tickets.iterrows():
            start, end = bounds[row["vehicle_id"]]
            assert start <= ts.iloc[i] <= end

    def test_reasons_from_config(self, small_config, tickets):
        assert set(tickets["reason"]).issubset(set(small_config.maintenance_reasons))


# ----------------------------------------------------------------------
# Attack injector / command stream
# ----------------------------------------------------------------------
class TestAttackInjector:
    def test_commands_generated(self, commands):
        assert len(commands) > 0

    def test_command_ids_unique(self, commands):
        assert commands["command_id"].is_unique

    def test_legit_commands_have_ticket_id(self, commands):
        legit = commands[commands["is_attack"] == False]  # noqa: E712
        assert legit["ticket_id"].notnull().all()

    def test_attack_commands_have_no_ticket(self, commands):
        """This is the core ground-truth invariant: injected attacks must
        never have a matching maintenance ticket, or the whole premise
        of the detector (no-ticket = suspicious) breaks."""
        attacks = commands[commands["is_attack"] == True]  # noqa: E712
        assert attacks["ticket_id"].isnull().all()

    def test_legit_command_count_matches_tickets(self, commands, tickets):
        legit = commands[commands["is_attack"] == False]  # noqa: E712
        assert len(legit) == len(tickets)

    def test_at_least_one_attack_injected(self, commands):
        """With attack_injection_rate_pct > 0 and a non-trivial fleet size,
        we expect at least one attack in the stream."""
        assert (commands["is_attack"] == True).sum() > 0  # noqa: E712

    def test_command_types_valid(self, small_config, commands):
        assert set(commands["command_type"]).issubset(set(small_config.command_types))

    def test_attack_command_types_valid(self, small_config, commands):
        attacks = commands[commands["is_attack"] == True]  # noqa: E712
        assert set(attacks["command_type"]).issubset(set(small_config.attack_command_types))

    def test_gps_coordinates_plausible(self, commands):
        assert commands["latitude"].between(-90, 90).all()
        assert commands["longitude"].between(-180, 180).all()

    def test_stream_sorted_by_timestamp(self, commands):
        ts = pd.to_datetime(commands["timestamp"])
        assert ts.is_monotonic_increasing


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def test_same_seed_produces_same_output(small_config):
    gen1 = TelemetryGenerator(small_config)
    gen2 = TelemetryGenerator(small_config)
    df1 = gen1.generate_fleet()
    df2 = gen2.generate_fleet()
    pd.testing.assert_frame_equal(df1, df2)