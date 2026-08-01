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
        self.rng = np.random.default_rng(config.random_seed + 3)
        self._driver_pool: List[dict] = self._generate_driver_pool()

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
        return list(self._driver_pool)

    def generate_vehicle_assignments(
        self, vehicle_id: str, time_bounds: Tuple[datetime, datetime]
    ) -> List[dict]:
        cfg = self.config
        start, end = time_bounds
        n_shifts = int(self.rng.integers(
            cfg.driver_shifts_per_vehicle[0], cfg.driver_shifts_per_vehicle[1] + 1,
        ))

        total_seconds = max((end - start).total_seconds(), 1.0)
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