from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig, default_config


class AttackInjector:
    def __init__(self, config: SimulatorConfig = default_config):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed + 2)  # offset from telemetry/maintenance

    # Legitimate commands — one per maintenance ticket
    def _legitimate_commands_from_tickets(self, tickets_df: pd.DataFrame) -> List[dict]:
        cfg = self.config
        legit_types = [c for c in cfg.command_types]  # all types can occur legitimately
        commands = []

        for _, ticket in tickets_df.iterrows():
            cmd_type = legit_types[self.rng.integers(0, len(legit_types))]
            # Legit commands happen right around the ticket time and at the ticket's location
            jitter_minutes = self.rng.uniform(-15, 15)
            ts = pd.to_datetime(ticket["timestamp"]) + timedelta(minutes=jitter_minutes)

            commands.append({
                "command_id": f"CMD-{uuid.uuid4().hex[:8].upper()}",
                "vehicle_id": ticket["vehicle_id"],
                "timestamp": ts.isoformat(),
                "command_type": cmd_type,
                "latitude": ticket["depot_lat"],
                "longitude": ticket["depot_lon"],
                "ticket_id": ticket["ticket_id"],
                "is_attack": False,
            })
        return commands

    # Injected attack commands — Tirri Challenge mechanics
    def _random_road_coords(self) -> Tuple[float, float]:
        """Picks a depot as a rough regional anchor, then jitters far enough
        away to represent 'in motion on a public road', not at the depot."""
        cfg = self.config
        depot = cfg.depot_locations[self.rng.integers(0, len(cfg.depot_locations))]
        lat = depot[0] + self.rng.uniform(-cfg.attack_road_gps_jitter_deg, cfg.attack_road_gps_jitter_deg)
        lon = depot[1] + self.rng.uniform(-cfg.attack_road_gps_jitter_deg, cfg.attack_road_gps_jitter_deg)
        return round(lat, 6), round(lon, 6)

    def _single_attack_command(self, vehicle_id: str, ts: datetime) -> dict:
        cfg = self.config
        cmd_type = cfg.attack_command_types[self.rng.integers(0, len(cfg.attack_command_types))]

        if self.rng.random() < cfg.attack_gps_mismatch_probability:
            lat, lon = self._random_road_coords()
        else:
            # Rare edge case: attack near a depot but still with no ticket —
            # keeps the detector honest, since GPS match alone isn't sufficient.
            depot = cfg.depot_locations[self.rng.integers(0, len(cfg.depot_locations))]
            lat, lon = round(depot[0], 6), round(depot[1], 6)

        return {
            "command_id": f"CMD-{uuid.uuid4().hex[:8].upper()}",
            "vehicle_id": vehicle_id,
            "timestamp": ts.isoformat(),
            "command_type": cmd_type,
            "latitude": lat,
            "longitude": lon,
            "ticket_id": None,
            "is_attack": True,
        }

    def _generate_vehicle_attacks(
        self, vehicle_id: str, time_bounds: Tuple[datetime, datetime]
    ) -> List[dict]:
        cfg = self.config
        start, end = time_bounds
        n_attacks = int(self.rng.integers(
            cfg.attacks_per_affected_vehicle[0],
            cfg.attacks_per_affected_vehicle[1] + 1,
        ))

        attacks = []
        for _ in range(n_attacks):
            delta_seconds = max((end - start).total_seconds(), 1)
            base_ts = start + timedelta(seconds=float(self.rng.uniform(0, delta_seconds)))

            if self.rng.random() < cfg.attack_frequency_burst_probability:
                # Frequency-spike attack: several commands fired within a short window,
                # mirroring pranksters repeatedly toggling the app.
                burst_count = int(self.rng.integers(
                    cfg.attack_burst_command_count[0],
                    cfg.attack_burst_command_count[1] + 1,
                ))
                for _ in range(burst_count):
                    offset = self.rng.uniform(0, cfg.attack_burst_window_seconds)
                    ts = base_ts + timedelta(seconds=float(offset))
                    attacks.append(self._single_attack_command(vehicle_id, ts))
            else:
                attacks.append(self._single_attack_command(vehicle_id, base_ts))

        return attacks

    def _select_affected_vehicles(self, vehicle_ids: List[str]) -> List[str]:
        cfg = self.config
        n_affected = max(1, int(round(len(vehicle_ids) * cfg.attack_injection_rate_pct)))
        chosen = self.rng.choice(vehicle_ids, size=n_affected, replace=False)
        return list(chosen)

    # Public entrypoint

    def generate_command_stream(
        self,
        vehicle_time_bounds: Dict[str, Tuple[datetime, datetime]],
        tickets_df: pd.DataFrame,
    ) -> pd.DataFrame:
        legit_commands = self._legitimate_commands_from_tickets(tickets_df)

        affected_vehicles = self._select_affected_vehicles(list(vehicle_time_bounds.keys()))
        attack_commands: List[dict] = []
        for vid in affected_vehicles:
            attack_commands.extend(self._generate_vehicle_attacks(vid, vehicle_time_bounds[vid]))

        all_commands = legit_commands + attack_commands
        df = pd.DataFrame(all_commands)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)

        n_attack = int(df["is_attack"].sum()) if not df.empty else 0
        print(f"[attack_injector] {len(affected_vehicles)} vehicle(s) affected, "
            f"{n_attack} attack command(s) injected out of {len(df)} total commands.")
        return df


if __name__ == "__main__":
    # Standalone end-to-end run: telemetry -> tickets -> command stream,
    # then dump all three to data/seed/ as the demo-safety-net CSVs.
    import os

    from .telemetry_generator import TelemetryGenerator
    from .maintenance_generator import MaintenanceGenerator

    cfg = default_config

    tgen = TelemetryGenerator(cfg)
    fleet_telemetry = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(fleet_telemetry)

    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    out_dir = cfg.seed_dir
    os.makedirs(out_dir, exist_ok=True)
    fleet_telemetry.to_csv(os.path.join(out_dir, cfg.seed_telemetry_filename), index=False)
    commands_df.to_csv(os.path.join(out_dir, cfg.seed_attacks_filename), index=False)

    print(f"\nWrote telemetry -> {out_dir}/{cfg.seed_telemetry_filename} "
        f"({len(fleet_telemetry)} rows)")
    print(f"Wrote command stream -> {out_dir}/{cfg.seed_attacks_filename} "
        f"({len(commands_df)} rows)")