"""
simulator/driver_generator.py

Generates a mock driver pool and per-vehicle shift-assignment records —
Future Roadmap Feature 1 (Driver Identity & Vehicle Assignment). Every
signal computed elsewhere in models/ (RUL, thermal anomalies, security
severity, charging stress) stays keyed on vehicle_id only; this module
exists purely to attach a "who was driving this vehicle, and when"
dimension on top, so a future driver-level aggregation (Feature 5,
"Driver-Level Coaching") has something real to group by.

Depends only on config.py and a vehicle_id / time-bounds input — the
same dependency shape as maintenance_generator.py — not on telemetry
values themselves.

Mechanics:
  1. A driver pool is generated ONCE for the whole fleet
     (cfg.num_drivers), independent of fleet_size, since in a real
     fleet drivers rotate across vehicles/shifts rather than each
     owning exactly one vehicle.
  2. Vehicle assignments are then generated per vehicle by carving its
     active time bound (from TelemetryGenerator.get_vehicle_time_bounds)
     into cfg.driver_shifts_per_vehicle contiguous, NON-overlapping
     shift windows, each randomly assigned a driver from the pool. Two
     consecutive assignments for the same vehicle can never overlap by
     construction (they're literally adjacent slices of the same
     window), which is the property tests/test_driver_assignments.py
     checks directly.

is_attack/thermal_event_flag-style ground-truth concerns don't apply
here — there's no "real" answer this needs to be validated against;
it's assignment data by definition, not something to detect.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SimulatorConfig, default_config


class DriverGenerator:
    def __init__(self, config: SimulatorConfig = default_config):
        self.config = config
        # Offset from telemetry (seed+0), maintenance (seed+1), and attack
        # injection (seed+2) generators, so this generator's draws are
        # reproducible and independent of how many random calls those made.
        self.rng = np.random.default_rng(config.random_seed + 3)
        self._driver_pool: List[dict] = self._generate_driver_pool()

    # ------------------------------------------------------------------
    def _generate_driver_pool(self) -> List[dict]:
        cfg = self.config
        depot_names = [f"Depot {i + 1}" for i in range(len(cfg.depot_locations))]
        drivers = []
        for i in range(cfg.num_drivers):
            first = cfg.driver_first_names[self.rng.integers(0, len(cfg.driver_first_names))]
            last = cfg.driver_last_names[self.rng.integers(0, len(cfg.driver_last_names))]
            depot = depot_names[self.rng.integers(0, len(depot_names))]
            drivers.append({
                "driver_id": f"DRV-{i + 1:04d}",
                "name": f"{first} {last}",
                "license_id": f"{cfg.license_id_prefix}-{int(self.rng.integers(100000, 999999))}",
                "depot_home": depot,
            })
        return drivers

    def get_driver_pool(self) -> List[dict]:
        """The generated driver pool as plain dicts — one row per driver,
        independent of any vehicle. Callers insert these once per seed
        run (not once per vehicle); ingestion/db.py's INSERT OR IGNORE
        makes re-insertion across repeated seed runs idempotent anyway."""
        return list(self._driver_pool)

    # ------------------------------------------------------------------
    def generate_vehicle_assignments(
        self, vehicle_id: str, time_bounds: Tuple[datetime, datetime]
    ) -> List[dict]:
        """Carves the vehicle's active window into N contiguous,
        non-overlapping shift-assignment records, each with a randomly
        picked driver from the pool.

        Every assignment gets a concrete shift_end here — the simulator
        only ever produces *historical* windows, so there's no
        "ongoing, end unknown yet" case to model (that's a real-
        ingestion-only scenario; see VehicleAssignment.shift_end's
        Optional-ness in ingestion/schemas.py)."""
        cfg = self.config
        start, end = time_bounds
        n_shifts = int(self.rng.integers(
            cfg.driver_shifts_per_vehicle[0], cfg.driver_shifts_per_vehicle[1] + 1,
        ))

        total_seconds = max((end - start).total_seconds(), 1.0)
        # Random cut points inside (start, end) split the window into
        # n_shifts contiguous, non-overlapping pieces — sorting the cut
        # fractions first is what guarantees adjacency (no gaps, no
        # overlaps) between consecutive assignments.
        cut_fracs = sorted(self.rng.uniform(0, 1, size=max(n_shifts - 1, 0)))
        boundaries = (
            [start]
            + [start + pd.Timedelta(seconds=frac * total_seconds).to_pytimedelta() for frac in cut_fracs]
            + [end]
        )

        assignments = []
        for i in range(len(boundaries) - 1):
            shift_start, shift_end = boundaries[i], boundaries[i + 1]
            driver = self._driver_pool[self.rng.integers(0, len(self._driver_pool))]
            assignments.append({
                "assignment_id": f"ASG-{uuid.uuid4().hex[:8].upper()}",
                "vehicle_id": vehicle_id,
                "driver_id": driver["driver_id"],
                "shift_start": shift_start.isoformat(),
                "shift_end": shift_end.isoformat(),
            })
        return assignments

    def generate_fleet_assignments(
        self, vehicle_time_bounds: Dict[str, Tuple[datetime, datetime]]
    ) -> pd.DataFrame:
        """vehicle_time_bounds: {vehicle_id: (first_ts, last_ts)}, typically
        from TelemetryGenerator.get_vehicle_time_bounds(fleet_telemetry_df)."""
        all_assignments = []
        for vid, bounds in vehicle_time_bounds.items():
            all_assignments.extend(self.generate_vehicle_assignments(vid, bounds))

        df = pd.DataFrame(all_assignments)
        if not df.empty:
            df = df.sort_values("shift_start").reset_index(drop=True)
        return df


if __name__ == "__main__":
    # Standalone sanity check — requires telemetry_generator to build time bounds.
    from simulator.telemetry_generator import TelemetryGenerator

    tgen = TelemetryGenerator()
    fleet_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(fleet_df)

    dgen = DriverGenerator()
    pool = dgen.get_driver_pool()
    assignments_df = dgen.generate_fleet_assignments(bounds)

    print(f"Generated {len(pool)} drivers and {len(assignments_df)} shift assignments "
        f"across {len(bounds)} vehicles.")
    print(pd.DataFrame(pool).head())
    print(assignments_df.head())