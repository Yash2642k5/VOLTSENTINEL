from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import SimulatorConfig, default_config


class MaintenanceGenerator:
    def __init__(self, config: SimulatorConfig = default_config):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed + 1)  # offset seed from telemetry

    def _jittered_depot_coords(self, depot: Tuple[float, float]) -> Tuple[float, float]:
        cfg = self.config
        lat, lon = depot
        lat += self.rng.uniform(-cfg.depot_gps_jitter_deg, cfg.depot_gps_jitter_deg)
        lon += self.rng.uniform(-cfg.depot_gps_jitter_deg, cfg.depot_gps_jitter_deg)
        return round(lat, 6), round(lon, 6)

    def _random_timestamp_in_range(self, start: datetime, end: datetime) -> datetime:
        if end <= start:
            return start
        delta_seconds = (end - start).total_seconds()
        offset = self.rng.uniform(0, delta_seconds)
        return start + pd.Timedelta(seconds=offset).to_pytimedelta()

    def generate_vehicle_tickets(
        self, vehicle_id: str, time_bounds: Tuple[datetime, datetime]
    ) -> List[dict]:
        cfg = self.config
        start, end = time_bounds
        n_tickets = int(self.rng.integers(
            cfg.maintenance_events_per_vehicle[0],
            cfg.maintenance_events_per_vehicle[1] + 1,
        ))

        tickets = []
        for _ in range(n_tickets):
            depot = cfg.depot_locations[self.rng.integers(0, len(cfg.depot_locations))]
            lat, lon = self._jittered_depot_coords(depot)
            ts = self._random_timestamp_in_range(start, end)
            reason = cfg.maintenance_reasons[self.rng.integers(0, len(cfg.maintenance_reasons))]

            tickets.append({
                "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
                "vehicle_id": vehicle_id,
                "timestamp": ts.isoformat(),
                "depot_lat": lat,
                "depot_lon": lon,
                "reason": reason,
                "technician_id": f"TECH-{int(self.rng.integers(100, 999))}",
            })
        return tickets

    def generate_fleet_tickets(
        self, vehicle_time_bounds: Dict[str, Tuple[datetime, datetime]]
    ) -> pd.DataFrame:
        all_tickets = []
        for vid, bounds in vehicle_time_bounds.items():
            all_tickets.extend(self.generate_vehicle_tickets(vid, bounds))

        df = pd.DataFrame(all_tickets)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df


if __name__ == "__main__":
    from simulator.telemetry_generator import TelemetryGenerator

    tgen = TelemetryGenerator()
    fleet_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(fleet_df)

    mgen = MaintenanceGenerator()
    tickets_df = mgen.generate_fleet_tickets(bounds)

    print(f"Generated {len(tickets_df)} maintenance tickets for {len(bounds)} vehicles.")
    print(tickets_df.head())