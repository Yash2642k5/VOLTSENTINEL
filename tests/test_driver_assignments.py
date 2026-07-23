"""
tests/test_driver_assignments.py

Validates Future Roadmap Feature 1 (Driver Identity & Vehicle
Assignment): simulator/driver_generator.py's pool/assignment
generation, the new Driver/VehicleAssignment pydantic schemas, and
ingestion/db.py's drivers/vehicle_assignments tables + insert/query
helpers — mirroring the existing style of tests/test_simulator.py and
tests/test_ingestion.py.

Run from the project root:
    pytest tests/test_driver_assignments.py -v
"""

import os

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.driver_generator import DriverGenerator

from ingestion.schemas import Driver, VehicleAssignment
from ingestion.db import (
    get_connection,
    init_db,
    insert_driver_batch,
    insert_vehicle_assignment_batch,
    get_all_drivers,
    get_driver,
    get_assignments_for_vehicle,
    get_assignments_for_driver,
    get_current_assignment_for_vehicle,
    get_current_driver_for_vehicle,
    row_counts,
)

TEST_DB_PATH = os.path.join("data", "test_driver_assignments.db")


@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=6, num_cycles=40, random_seed=17, num_drivers=5)


@pytest.fixture(scope="module")
def telemetry(small_config):
    gen = TelemetryGenerator(small_config)
    return gen, gen.generate_fleet()


@pytest.fixture(scope="module")
def bounds(telemetry):
    gen, df = telemetry
    return gen.get_vehicle_time_bounds(df)


@pytest.fixture(scope="module")
def dgen(small_config):
    return DriverGenerator(small_config)


@pytest.fixture(scope="module")
def driver_pool(dgen):
    return dgen.get_driver_pool()


@pytest.fixture(scope="module")
def assignments_df(dgen, bounds):
    return dgen.generate_fleet_assignments(bounds)


# ----------------------------------------------------------------------
# DriverGenerator — pure, no DB
# ----------------------------------------------------------------------
class TestDriverPool:
    def test_pool_size_matches_config(self, small_config, driver_pool):
        assert len(driver_pool) == small_config.num_drivers

    def test_driver_ids_unique(self, driver_pool):
        ids = [d["driver_id"] for d in driver_pool]
        assert len(ids) == len(set(ids))

    def test_every_driver_has_required_fields(self, driver_pool):
        for d in driver_pool:
            assert d["driver_id"] and d["name"] and d["license_id"] and d["depot_home"]

    def test_same_seed_produces_same_pool(self, small_config):
        pool_a = DriverGenerator(small_config).get_driver_pool()
        pool_b = DriverGenerator(small_config).get_driver_pool()
        assert pool_a == pool_b


class TestVehicleAssignments:
    def test_assignments_generated_for_every_vehicle(self, small_config, assignments_df):
        expected_vehicles = {f"EVR-{i:04d}" for i in range(1, small_config.fleet_size + 1)}
        assert set(assignments_df["vehicle_id"]) == expected_vehicles

    def test_assignment_ids_unique(self, assignments_df):
        assert assignments_df["assignment_id"].is_unique

    def test_assignments_reference_pool_drivers(self, assignments_df, driver_pool):
        pool_ids = {d["driver_id"] for d in driver_pool}
        assert set(assignments_df["driver_id"]).issubset(pool_ids)

    def test_shifts_within_vehicle_time_bounds(self, bounds, assignments_df):
        for _, row in assignments_df.iterrows():
            start, end = bounds[row["vehicle_id"]]
            s = pd.to_datetime(row["shift_start"])
            e = pd.to_datetime(row["shift_end"])
            assert start <= s <= e <= end

    def test_shifts_do_not_overlap_within_a_vehicle(self, assignments_df):
        """The core correctness property: consecutive shift-assignment
        records for the same vehicle must be contiguous and non-
        overlapping, since a vehicle can't have two drivers at once."""
        for vid, group in assignments_df.groupby("vehicle_id"):
            group = group.sort_values("shift_start")
            starts = pd.to_datetime(group["shift_start"]).tolist()
            ends = pd.to_datetime(group["shift_end"]).tolist()
            for i in range(1, len(starts)):
                assert starts[i] >= ends[i - 1]

    def test_shift_count_within_configured_range(self, small_config, assignments_df):
        lo, hi = small_config.driver_shifts_per_vehicle
        counts = assignments_df.groupby("vehicle_id").size()
        assert counts.between(lo, hi).all()


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class TestSchemas:
    def test_driver_validates(self, driver_pool):
        for d in driver_pool:
            Driver(**d)

    def test_assignment_validates(self, assignments_df):
        for row in assignments_df.to_dict(orient="records"):
            VehicleAssignment(**row)

    def test_assignment_shift_end_is_optional(self):
        """The simulator always produces a concrete shift_end, but real
        (live) ingestion should be able to accept an in-progress shift
        without one yet."""
        VehicleAssignment(
            assignment_id="ASG-1", vehicle_id="EVR-0001", driver_id="DRV-0001",
            shift_start="2026-01-01T00:00:00",
        )

    def test_driver_extra_field_rejected(self):
        with pytest.raises(Exception):
            Driver(
                driver_id="DRV-0001", name="Test Driver", license_id="DL-100000",
                depot_home="Depot 1", made_up_field="should not be allowed",
            )

    def test_assignment_extra_field_rejected(self):
        with pytest.raises(Exception):
            VehicleAssignment(
                assignment_id="ASG-1", vehicle_id="EVR-0001", driver_id="DRV-0001",
                shift_start="2026-01-01T00:00:00", made_up_field="nope",
            )

    def test_driver_empty_id_rejected(self):
        with pytest.raises(Exception):
            Driver(driver_id="   ", name="Test", license_id="DL-1", depot_home="Depot 1")


# ----------------------------------------------------------------------
# DB round trip
# ----------------------------------------------------------------------
@pytest.fixture
def conn(driver_pool, assignments_df):
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)

    drivers = [Driver(**d) for d in driver_pool]
    assignments = [VehicleAssignment(**row) for row in assignments_df.to_dict(orient="records")]

    insert_driver_batch(connection, drivers)
    insert_vehicle_assignment_batch(connection, assignments)

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


class TestDatabase:
    def test_tables_created(self, conn):
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"drivers", "vehicle_assignments"}.issubset(tables)

    def test_row_counts_match_source(self, conn, driver_pool, assignments_df):
        counts = row_counts(conn)
        assert counts["drivers"] == len(driver_pool)
        assert counts["vehicle_assignments"] == len(assignments_df)

    def test_get_driver_by_id(self, conn, driver_pool):
        target = driver_pool[0]
        row = get_driver(conn, target["driver_id"])
        assert row is not None
        assert row["name"] == target["name"]
        assert row["license_id"] == target["license_id"]

    def test_get_driver_unknown_id_returns_none(self, conn):
        assert get_driver(conn, "DRV-9999") is None

    def test_get_assignments_for_vehicle_ordered_by_shift_start(self, conn, assignments_df):
        vid = assignments_df["vehicle_id"].iloc[0]
        rows = get_assignments_for_vehicle(conn, vid)
        starts = [r["shift_start"] for r in rows]
        assert starts == sorted(starts)
        assert all(r["vehicle_id"] == vid for r in rows)

    def test_get_assignments_for_driver(self, conn, driver_pool):
        target = driver_pool[0]
        rows = get_assignments_for_driver(conn, target["driver_id"])
        assert all(r["driver_id"] == target["driver_id"] for r in rows)

    def test_current_assignment_is_the_latest_shift(self, conn, assignments_df):
        vid = assignments_df["vehicle_id"].iloc[0]
        expected_latest = (
            assignments_df[assignments_df.vehicle_id == vid]
            .sort_values("shift_start")
            .iloc[-1]
        )
        current = get_current_assignment_for_vehicle(conn, vid)
        assert current is not None
        assert current["assignment_id"] == expected_latest["assignment_id"]

    def test_current_assignment_unknown_vehicle_returns_none(self, conn):
        assert get_current_assignment_for_vehicle(conn, "EVR-9999") is None

    def test_current_driver_for_vehicle_joins_correctly(self, conn, assignments_df, driver_pool):
        vid = assignments_df["vehicle_id"].iloc[0]
        result = get_current_driver_for_vehicle(conn, vid)
        assert result is not None
        assert result["vehicle_id"] == vid
        assert result["driver_id"] in {d["driver_id"] for d in driver_pool}
        assert result["name"]
        assert result["license_id"]

    def test_current_driver_unassigned_vehicle_returns_none(self, conn):
        assert get_current_driver_for_vehicle(conn, "EVR-9999") is None

    def test_reinsertion_is_idempotent(self, conn, driver_pool, assignments_df):
        drivers = [Driver(**d) for d in driver_pool]
        assignments = [VehicleAssignment(**row) for row in assignments_df.to_dict(orient="records")]

        before = row_counts(conn)
        insert_driver_batch(conn, drivers)
        insert_vehicle_assignment_batch(conn, assignments)
        after = row_counts(conn)

        assert before["drivers"] == after["drivers"]
        assert before["vehicle_assignments"] == after["vehicle_assignments"]

    def test_foreign_key_integrity(self, conn):
        """Every driver_id in vehicle_assignments must reference a real
        driver — catches a schema/generator drift the same way
        test_ingestion.py's test_foreign_key_integrity does for
        commands -> maintenance_tickets."""
        rows = conn.execute(
            """SELECT a.driver_id FROM vehicle_assignments a
                LEFT JOIN drivers d ON a.driver_id = d.driver_id
                WHERE d.driver_id IS NULL"""
        ).fetchall()
        assert len(rows) == 0