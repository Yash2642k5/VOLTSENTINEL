"""
Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --fleet-size 30 --cycles 300 --seed 42
"""

from __future__ import annotations

import argparse
import math
import os

from ingestion.db import (
    get_connection, init_db, insert_telemetry_batch,
    insert_maintenance_batch, insert_command_batch, row_counts,
    insert_driver_batch, insert_vehicle_assignment_batch,
    insert_vehicle_metadata_batch,
)
from ingestion.schemas import (
    TelemetryReading, MaintenanceTicket, CommandEvent, Driver, VehicleAssignment,
    VehicleMetadata,
)
from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector
from simulator.driver_generator import DriverGenerator
from simulator.asset_generator import AssetGenerator


def seed(fleet_size: int, num_cycles: int, random_seed: int, db_path: str, wipe: bool) -> None:
    if wipe and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing DB at {db_path}")

    cfg = SimulatorConfig(fleet_size=fleet_size, num_cycles=num_cycles, random_seed=random_seed)

    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)

    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    dgen = DriverGenerator(cfg)
    driver_pool = dgen.get_driver_pool()
    assignments_df = dgen.generate_fleet_assignments(bounds)

    agen = AssetGenerator(cfg)
    assets_df = agen.generate_fleet_assets(tgen.vehicle_ids)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**r) for r in tickets_df.to_dict(orient="records")]
    commands = []
    for r in commands_df.to_dict(orient="records"):
        if isinstance(r.get("ticket_id"), float) and math.isnan(r["ticket_id"]):
            r["ticket_id"] = None
        commands.append(CommandEvent(**r))
    drivers = [Driver(**r) for r in driver_pool]
    assignments = [VehicleAssignment(**r) for r in assignments_df.to_dict(orient="records")]
    assets = [VehicleMetadata(**r) for r in assets_df.to_dict(orient="records")]

    conn = get_connection(db_path)
    init_db(conn)

    insert_telemetry_batch(conn, readings)
    insert_maintenance_batch(conn, tickets)
    insert_command_batch(conn, commands)
    insert_driver_batch(conn, drivers)
    insert_vehicle_assignment_batch(conn, assignments)
    insert_vehicle_metadata_batch(conn, assets)

    print(f"Seeded {db_path} — {row_counts(conn)}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reseed VoltSentinel's SQLite DB")
    parser.add_argument("--fleet-size", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", default=os.path.join("data", "voltsentinel.db"))
    parser.add_argument("--no-wipe", action="store_true", help="Keep existing rows (relies on OR IGNORE dedup)")
    args = parser.parse_args()

    seed(args.fleet_size, args.cycles, args.seed, args.db_path, wipe=not args.no_wipe)