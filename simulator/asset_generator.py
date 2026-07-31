"""Generates static asset-registry metadata (make, model, VIN, purchase
date, warranty expiry) for the simulated fleet."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from .config import SimulatorConfig, default_config

_VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # excludes I, O, Q, like real VINs


class AssetGenerator:
    def __init__(self, config: SimulatorConfig = default_config):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed + 4)  # +4: after telemetry/maintenance/attack/driver

    def _generate_vin(self) -> str:
        return "".join(self.rng.choice(list(_VIN_ALPHABET), size=17))

    def generate_fleet_assets(self, vehicle_ids: List[str]) -> pd.DataFrame:
        cfg = self.config
        sim_start = datetime.fromisoformat(cfg.sim_start_date)
        rows = []
        for vid in vehicle_ids:
            make, model = cfg.vehicle_makes_models[
                self.rng.integers(0, len(cfg.vehicle_makes_models))
            ]
            age_days = int(self.rng.integers(
                cfg.vehicle_age_range_days[0], cfg.vehicle_age_range_days[1] + 1
            ))
            purchase_date = sim_start - timedelta(days=age_days)
            warranty_expiry_date = purchase_date + timedelta(days=int(cfg.vehicle_warranty_years * 365))
            rows.append({
                "vehicle_id": vid,
                "make": make,
                "model": model,
                "vin": self._generate_vin(),
                "purchase_date": purchase_date.isoformat(),
                "warranty_expiry_date": warranty_expiry_date.isoformat(),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from simulator.telemetry_generator import TelemetryGenerator

    tgen = TelemetryGenerator()
    agen = AssetGenerator()
    assets_df = agen.generate_fleet_assets(tgen.vehicle_ids)
    print(f"Generated asset-registry metadata for {len(assets_df)} vehicles.")
    print(assets_df.head())
