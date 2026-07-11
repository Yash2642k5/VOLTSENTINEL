"""
tests/test_ingestion.py

Validates Phase 2's storage layer so far: schemas.py (pydantic models)
and db.py (SQLite tables + insert/query helpers). routes.py and
main.py aren't built yet, so this only covers the DB boundary —
not the REST/WebSocket boundary.

Run from the project root:
    pytest tests/test_ingestion.py -v
"""

import math
import os

import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector

from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent
from ingestion import db as db_module


TEST_DB_PATH = os.path.join("data", "test_ingestion.db")


@pytest.fixture(scope="module")
def simulated_data():
    cfg = SimulatorConfig(fleet_size=6, num_cycles=25, random_seed=2026)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)

    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    return telem_df, tickets_df, commands_df


@pytest.fixture(scope="module")
def validated_models(simulated_data):
    """Every row from every simulator stream, parsed through the pydantic
    schemas. If this fixture fails, schemas.py doesn't match simulator output."""
    telem_df, tickets_df, commands_df = simulated_data

    readings = [TelemetryReading(**row) for row in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**row) for row in tickets_df.to_dict(orient="records")]

    commands = []
    for row in commands_df.to_dict(orient="records"):
        if isinstance(row.get("ticket_id"), float) and math.isnan(row["ticket_id"]):
            row["ticket_id"] = None
        commands.append(CommandEvent(**row))

    return readings, tickets, commands


@pytest.fixture
def conn(validated_models):
    """Fresh SQLite DB per test, loaded with the full validated dataset."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = db_module.get_connection(TEST_DB_PATH)
    db_module.init_db(connection)

    readings, tickets, commands = validated_models
    db_module.insert_telemetry_batch(connection, readings)
    db_module.insert_maintenance_batch(connection, tickets)
    db_module.insert_command_batch(connection, commands)

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


# ----------------------------------------------------------------------
# schemas.py — does every simulator row actually validate?
# ----------------------------------------------------------------------
class TestSchemas:
    def test_all_telemetry_rows_valid(self, simulated_data, validated_models):
        telem_df, _, _ = simulated_data
        readings, _, _ = validated_models
        assert len(readings) == len(telem_df)

    def test_all_ticket_rows_valid(self, simulated_data, validated_models):
        _, tickets_df, _ = simulated_data
        _, tickets, _ = validated_models
        assert len(tickets) == len(tickets_df)

    def test_all_command_rows_valid(self, simulated_data, validated_models):
        _, _, commands_df = simulated_data
        _, _, commands = validated_models
        assert len(commands) == len(commands_df)

    def test_extra_field_rejected(self):
        """model_config = {'extra': 'forbid'} should reject unknown fields —
        catches silent data-shape drift between simulator and schema."""
        with pytest.raises(Exception):
            TelemetryReading(
                vehicle_id="EVR-0001", cycle=1, timestamp="2026-01-01T00:00:00",
                capacity_kwh=3.0, capacity_pct_of_rated=95.0, rated_capacity_kwh=3.2,
                voltage=51.0, temperature_c=30.0, soc_pct=80.0, is_fast_charge=False,
                dod_pct=50.0, made_up_field="should not be allowed",
            )

    def test_soc_out_of_range_rejected(self):
        with pytest.raises(Exception):
            TelemetryReading(
                vehicle_id="EVR-0001", cycle=1, timestamp="2026-01-01T00:00:00",
                capacity_kwh=3.0, capacity_pct_of_rated=95.0, rated_capacity_kwh=3.2,
                voltage=51.0, temperature_c=30.0, soc_pct=150.0,  # invalid
                is_fast_charge=False, dod_pct=50.0,
            )

    def test_invalid_command_type_rejected(self):
        with pytest.raises(Exception):
            CommandEvent(
                command_id="CMD-1", vehicle_id="EVR-0001", timestamp="2026-01-01T00:00:00",
                command_type="not_a_real_type", latitude=12.9, longitude=77.5,
            )


# ----------------------------------------------------------------------
# db.py — table creation, inserts, round-trip integrity
# ----------------------------------------------------------------------
class TestDatabase:
    def test_tables_created(self, conn):
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"telemetry", "maintenance_tickets", "commands"}.issubset(tables)

    def test_row_counts_match_source(self, conn, simulated_data):
        telem_df, tickets_df, commands_df = simulated_data
        counts = db_module.row_counts(conn)
        assert counts["telemetry"] == len(telem_df)
        assert counts["maintenance_tickets"] == len(tickets_df)
        assert counts["commands"] == len(commands_df)

    def test_vehicle_ids_roundtrip(self, conn, simulated_data):
        telem_df, _, _ = simulated_data
        expected = set(telem_df["vehicle_id"].unique())
        actual = set(db_module.get_all_vehicle_ids(conn))
        assert actual == expected

    def test_telemetry_ordered_by_cycle(self, conn, simulated_data):
        telem_df, _, _ = simulated_data
        vid = telem_df["vehicle_id"].iloc[0]
        rows = db_module.get_telemetry_for_vehicle(conn, vid)
        cycles = [r["cycle"] for r in rows]
        assert cycles == sorted(cycles)

    def test_unticketed_commands_match_attack_count(self, conn, simulated_data):
        """The core invariant: DB-side 'no ticket' must equal simulator-side
        'is_attack' count, or the security-anomaly precondition breaks."""
        _, _, commands_df = simulated_data
        expected_attacks = int(commands_df["is_attack"].sum())
        actual_unticketed = len(db_module.get_unticketed_commands(conn))
        assert actual_unticketed == expected_attacks

    def test_foreign_key_integrity(self, conn):
        """Every non-null ticket_id in commands must reference a real ticket."""
        rows = conn.execute(
            """SELECT c.ticket_id FROM commands c
                LEFT JOIN maintenance_tickets t ON c.ticket_id = t.ticket_id
                WHERE c.ticket_id IS NOT NULL AND t.ticket_id IS NULL"""
        ).fetchall()
        assert len(rows) == 0

    def test_reinsertion_is_idempotent(self, conn, validated_models):
        before = db_module.row_counts(conn)
        readings, tickets, commands = validated_models
        db_module.insert_telemetry_batch(conn, readings)
        db_module.insert_maintenance_batch(conn, tickets)
        db_module.insert_command_batch(conn, commands)
        after = db_module.row_counts(conn)
        assert before == after

    def test_telemetry_unique_constraint(self, conn):
        """(vehicle_id, cycle) must be unique — duplicate cycle for the same
        vehicle should be silently ignored, not inserted as a second row."""
        dup = TelemetryReading(
            vehicle_id="EVR-0001", cycle=1, timestamp="2026-01-01T00:00:00",
            capacity_kwh=999, capacity_pct_of_rated=100, rated_capacity_kwh=999,
            voltage=51.0, temperature_c=30.0, soc_pct=80.0, is_fast_charge=False, dod_pct=50.0,
        )
        before = db_module.row_counts(conn)["telemetry"]
        db_module.insert_telemetry(conn, dup)
        conn.commit()
        after = db_module.row_counts(conn)["telemetry"]
        assert before == after