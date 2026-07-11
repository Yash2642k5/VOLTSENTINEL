"""
simulator/telemetry_generator.py

Generates per-vehicle battery telemetry for the whole simulated fleet:
capacity fade (exponential decay + noise), voltage, temperature, SoC,
and per-cycle charging behaviour (fast-charge flag, depth-of-discharge).

Depends only on config.py. Does not know about maintenance tickets or
BMS command/attack events — those are separate streams produced by
maintenance_generator.py and attack_injector.py, which key off the
vehicle IDs and timestamps produced here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import SimulatorConfig, default_config


class TelemetryGenerator:
    def __init__(self, config: SimulatorConfig = default_config):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.vehicle_ids = [f"EVR-{i:04d}" for i in range(1, config.fleet_size + 1)]

        # Per-vehicle manufacturing/degradation variance, fixed once per vehicle
        # so repeated calls for the same vehicle are internally consistent.
        self._decay_rates: Dict[str, float] = {}
        self._rated_capacities: Dict[str, float] = {}
        self._init_vehicle_params()

    def _init_vehicle_params(self) -> None:
        cfg = self.config
        for vid in self.vehicle_ids:
            decay = self.rng.normal(cfg.decay_rate_mean, cfg.decay_rate_std)
            self._decay_rates[vid] = max(decay, cfg.decay_rate_mean * 0.1)

            variance = self.rng.uniform(-cfg.capacity_variance_pct, cfg.capacity_variance_pct)
            self._rated_capacities[vid] = cfg.rated_capacity_kwh * (1 + variance)

    def _generate_timestamps(self, num_cycles: int) -> list[datetime]:
        """Spaces cycles out realistically instead of one-per-day mechanically —
        adds jitter around the configured average cycles/day."""
        cfg = self.config
        start = datetime.fromisoformat(cfg.sim_start_date)
        avg_gap_hours = 24.0 / max(cfg.avg_cycles_per_day, 0.01)

        timestamps = []
        current = start
        for _ in range(num_cycles):
            jitter_hours = self.rng.normal(0, avg_gap_hours * 0.25)
            current = current + timedelta(hours=max(avg_gap_hours + jitter_hours, 1))
            timestamps.append(current)
        return timestamps

    def generate_vehicle_telemetry(self, vehicle_id: str) -> pd.DataFrame:
        cfg = self.config
        decay_rate = self._decay_rates[vehicle_id]
        rated_capacity = self._rated_capacities[vehicle_id]
        timestamps = self._generate_timestamps(cfg.num_cycles)

        rows = []
        stress_cycles = 0.0  # effective cycle count, inflated by fast-charge/high-DoD stress

        for cycle, ts in enumerate(timestamps, start=1):
            # --- Charging behaviour for this cycle ---
            is_fast_charge = self.rng.random() < cfg.fast_charge_probability
            dod_pct = float(np.clip(self.rng.normal(cfg.dod_mean_pct, cfg.dod_std_pct), 10, 100))
            is_high_dod = dod_pct >= cfg.high_dod_threshold_pct

            stress_multiplier = 1.0
            if is_fast_charge:
                stress_multiplier *= cfg.fast_charge_stress_multiplier
            if is_high_dod:
                stress_multiplier *= 1.3
            stress_cycles += stress_multiplier

            # --- Capacity fade (exponential decay against accumulated stress) ---
            capacity_noise = self.rng.normal(0, rated_capacity * cfg.capacity_noise_std_pct)
            capacity_kwh = rated_capacity * np.exp(-decay_rate * stress_cycles) + capacity_noise
            capacity_kwh = max(capacity_kwh, rated_capacity * 0.3)  # floor, avoid nonsense values
            capacity_pct_of_rated = round(100 * capacity_kwh / rated_capacity, 2)

            # --- Thermal behaviour ---
            temp = self.rng.normal(cfg.ambient_temp_mean_c, cfg.ambient_temp_std_c)
            if is_fast_charge:
                temp += cfg.charging_temp_rise_c
            thermal_event_flag = False
            if self.rng.random() < cfg.thermal_event_probability:
                temp += self.rng.uniform(10, 20)  # excursion above safe threshold
                thermal_event_flag = temp >= cfg.safe_temp_threshold_c
            temperature_c = round(float(temp), 2)

            # --- Voltage / SoC ---
            soc_end_pct = float(np.clip(100 - dod_pct + self.rng.normal(0, 3),
                                         cfg.soc_min_pct, cfg.soc_max_pct))
            voltage = round(float(
                cfg.nominal_voltage * (0.9 + 0.1 * soc_end_pct / 100)
                + self.rng.normal(0, cfg.voltage_noise_std)
            ), 2)

            rows.append({
                "vehicle_id": vehicle_id,
                "cycle": cycle,
                "timestamp": ts.isoformat(),
                "capacity_kwh": round(float(capacity_kwh), 3),
                "capacity_pct_of_rated": capacity_pct_of_rated,
                "rated_capacity_kwh": round(rated_capacity, 3),
                "voltage": voltage,
                "temperature_c": temperature_c,
                "soc_pct": round(soc_end_pct, 2),
                "is_fast_charge": is_fast_charge,
                "dod_pct": round(dod_pct, 2),
                "thermal_event_flag": thermal_event_flag,  # ground truth, for test validation only
            })

        return pd.DataFrame(rows)

    def generate_fleet(self) -> pd.DataFrame:
        """Generates telemetry for every vehicle in the fleet and concatenates it
        into a single DataFrame — this is what ingestion/db.py ultimately loads."""
        frames = [self.generate_vehicle_telemetry(vid) for vid in self.vehicle_ids]
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def get_vehicle_time_bounds(telemetry_df: pd.DataFrame) -> Dict[str, Tuple[datetime, datetime]]:
        """Returns {vehicle_id: (first_timestamp, last_timestamp)}.
        Used by maintenance_generator and attack_injector to place events
        within a vehicle's actual active window."""
        bounds = {}
        df = telemetry_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for vid, group in df.groupby("vehicle_id"):
            bounds[vid] = (group["timestamp"].min().to_pydatetime(),
                           group["timestamp"].max().to_pydatetime())
        return bounds


if __name__ == "__main__":
    # Standalone sanity check: generate telemetry and print a quick summary.
    gen = TelemetryGenerator()
    fleet_df = gen.generate_fleet()
    print(f"Generated {len(fleet_df)} telemetry rows for {gen.config.fleet_size} vehicles "
          f"across {gen.config.num_cycles} cycles.")
    print(fleet_df.head())
    print("\nCapacity fade sample (EVR-0001, every 100th cycle):")
    print(fleet_df[fleet_df.vehicle_id == "EVR-0001"].iloc[::100][
        ["cycle", "capacity_pct_of_rated", "temperature_c", "is_fast_charge", "dod_pct"]
    ])