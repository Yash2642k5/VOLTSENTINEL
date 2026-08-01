from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimulatorConfig, default_config
from .telemetry_generator import TelemetryGenerator

DEFAULT_MOVE_PROBABILITY = 0.85     # fraction of ticks a vehicle is actively driving
DEFAULT_STEP_DEG = 0.001            # ~100m per tick
DEFAULT_OPERATING_RADIUS_DEG = 0.15  # ~15km pull-back boundary around home depot
DEFAULT_INACTIVE_AFTER_SECONDS = 180.0  # 3 minutes with no movement -> inactive


class LiveTelemetryFeed:
    def __init__(
        self,
        config: SimulatorConfig = default_config,
        seed: Optional[int] = None,
        move_probability: float = DEFAULT_MOVE_PROBABILITY,
        step_deg: float = DEFAULT_STEP_DEG,
        operating_radius_deg: float = DEFAULT_OPERATING_RADIUS_DEG,
        inactive_after_seconds: float = DEFAULT_INACTIVE_AFTER_SECONDS,
    ):
        self.config = config
        self.tgen = TelemetryGenerator(config)
        self.rng = np.random.default_rng(seed)
        self.tgen.rng = self.rng  # generate_cycle_row draws from self.tgen.rng
        self.move_probability = move_probability
        self.step_deg = step_deg
        self.operating_radius_deg = operating_radius_deg
        self.inactive_after_seconds = inactive_after_seconds

        self._next_cycle: Dict[str, int] = {}
        self._stress_cycles: Dict[str, float] = {}
        self._home: Dict[str, Tuple[float, float]] = {
            vid: config.depot_locations[i % len(config.depot_locations)]
            for i, vid in enumerate(self.tgen.vehicle_ids)
        }
        self._position: Dict[str, Tuple[float, float]] = {}
        self._status: Dict[str, str] = {}
        self._last_moved_at: Dict[str, datetime] = {}

    @staticmethod
    def _estimate_stress_cycles(decay_rate: float, rated_capacity: float, capacity_kwh: float) -> float:
        ratio = max(capacity_kwh, 1e-6) / rated_capacity
        if ratio >= 1.0 or decay_rate <= 0:
            return 0.0
        return max(0.0, -math.log(ratio) / decay_rate)

    def prime_from_db(self, conn) -> None:
        from ingestion.db import get_latest_telemetry_for_vehicle, get_vehicle_live_state

        for vid in self.tgen.vehicle_ids:
            decay_rate, rated_capacity = self.tgen.vehicle_params(vid)
            latest = get_latest_telemetry_for_vehicle(conn, vid)
            if latest is None:
                self._next_cycle[vid] = 1
                self._stress_cycles[vid] = 0.0
            else:
                self._next_cycle[vid] = latest["cycle"] + 1
                self._stress_cycles[vid] = self._estimate_stress_cycles(
                    decay_rate, rated_capacity, latest["capacity_kwh"]
                )

            state = get_vehicle_live_state(conn, vid)
            if state is None:
                jitter = self.config.depot_gps_jitter_deg
                home = self._home[vid]
                self._position[vid] = (
                    home[0] + self.rng.uniform(-jitter, jitter),
                    home[1] + self.rng.uniform(-jitter, jitter),
                )
                self._status[vid] = "active"
                self._last_moved_at[vid] = datetime.now(timezone.utc)
            else:
                self._position[vid] = (state["latitude"], state["longitude"])
                self._status[vid] = state["status"]
                self._last_moved_at[vid] = datetime.fromisoformat(state["last_moved_at"])

    def _advance_position(self, vehicle_id: str, now: datetime) -> None:
        lat, lon = self._position[vehicle_id]
        if self.rng.random() < self.move_probability:
            dlat = self.rng.normal(0, self.step_deg)
            dlon = self.rng.normal(0, self.step_deg)
            home = self._home[vehicle_id]
            if math.hypot(lat - home[0], lon - home[1]) > self.operating_radius_deg:
                dlat += (home[0] - lat) * 0.15  # nudge back toward the operating area
                dlon += (home[1] - lon) * 0.15
            self._position[vehicle_id] = (lat + dlat, lon + dlon)
            self._last_moved_at[vehicle_id] = now
            self._status[vehicle_id] = "active"
        else:
            idle_seconds = (now - self._last_moved_at[vehicle_id]).total_seconds()
            self._status[vehicle_id] = "inactive" if idle_seconds > self.inactive_after_seconds else "active"

    def tick(self, conn) -> List[dict]:
        """Generates one new telemetry cycle and advances position/activity
        status for every vehicle."""
        from ingestion.db import insert_telemetry, upsert_vehicle_live_state
        from ingestion.schemas import TelemetryReading

        now = datetime.now(timezone.utc)
        inserted = []
        for vid in self.tgen.vehicle_ids:
            decay_rate, rated_capacity = self.tgen.vehicle_params(vid)
            cycle = self._next_cycle[vid]
            row, stress_cycles = self.tgen.generate_cycle_row(
                vid, cycle, now, decay_rate, rated_capacity, self._stress_cycles[vid]
            )
            self._stress_cycles[vid] = stress_cycles
            self._next_cycle[vid] = cycle + 1
            insert_telemetry(conn, TelemetryReading(**row))
            inserted.append(row)

            self._advance_position(vid, now)
            lat, lon = self._position[vid]
            upsert_vehicle_live_state(
                conn, vid, lat, lon, self._status[vid],
                self._last_moved_at[vid].isoformat(), now.isoformat(),
            )
        conn.commit()
        return inserted

    def run_forever(self, conn, interval_seconds: float = 15.0, max_ticks: Optional[int] = None) -> None:
        self.prime_from_db(conn)
        tick_count = 0
        while max_ticks is None or tick_count < max_ticks:
            inserted = self.tick(conn)
            tick_count += 1
            active_count = sum(1 for s in self._status.values() if s == "active")
            print(
                f"[live_feed] tick {tick_count}: inserted {len(inserted)} readings, "
                f"{active_count}/{len(self._status)} vehicles active, "
                f"at {datetime.now(timezone.utc).isoformat()}"
            )
            if max_ticks is None or tick_count < max_ticks:
                time.sleep(interval_seconds)
