"""
models/range_estimator.py

Future Roadmap Feature 2 — Live SoC / Range Tile.

Answers the fleet manager's most operationally urgent question: "can
every vehicle currently out on a route make it back without stopping
to charge, right now?" Distinct from rul_model.py's long-horizon
capacity-fade projection — this is a same-day, current-moment signal
built from each vehicle's LATEST telemetry reading only, not its full
history.

    estimated_range_km = capacity_kwh_remaining / kwh_per_km

  - capacity_kwh_remaining: the vehicle's current full-charge capacity
    (already faded — latest telemetry row's capacity_kwh) times its
    current state of charge (soc_pct). This is how much energy is
    actually in the pack right now, not the nameplate/rated figure.
  - kwh_per_km: no per-cycle distance/odometer field exists anywhere in
    this system's telemetry today (simulator/telemetry_generator.py
    never modeled one), so a real *historical, per-vehicle* kWh/km
    can't be derived from real data yet. This uses a single, explicit,
    documented fleet-wide constant
    (simulator.config.SimulatorConfig.avg_kwh_per_km) instead — the
    same "simple, explainable constant over a fabricated black-box
    number" choice rul_model.py already makes for
    end_of_life_capacity_pct. estimate_vehicle()/estimate_fleet() both
    accept an optional per-vehicle kwh_per_km override, so a future
    telemetry_generator extension with real odometer/GPS-distance data
    can swap in an actual historical average without an API change here.

Independent of simulator/config.py by design (same reasoning as
rul_model.py's DEFAULT_END_OF_LIFE_CAPACITY_PCT): the default constants
below mirror simulator/config.py's defaults so this works out of the box
against simulated data, but stay plain parameters here, not an import —
models/ should work against whatever is in SQLite regardless of source.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

DEFAULT_KWH_PER_KM = 0.06                # e-rickshaw-class energy consumption
DEFAULT_LOW_RANGE_THRESHOLD_KM = 15.0    # below this = "at risk of stranding"
DEFAULT_LOW_SOC_THRESHOLD_PCT = 20.0     # belt-and-braces SoC-only check


@dataclass
class RangeEstimate:
    vehicle_id: str
    latest_cycle: Optional[int]
    soc_pct: Optional[float]
    capacity_kwh_remaining: Optional[float]
    kwh_per_km: float
    estimated_range_km: Optional[float]
    at_risk_of_stranding: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RangeEstimator:
    def __init__(
        self,
        kwh_per_km: float = DEFAULT_KWH_PER_KM,
        low_range_threshold_km: float = DEFAULT_LOW_RANGE_THRESHOLD_KM,
        low_soc_threshold_pct: float = DEFAULT_LOW_SOC_THRESHOLD_PCT,
    ):
        self.kwh_per_km = kwh_per_km
        self.low_range_threshold_km = low_range_threshold_km
        self.low_soc_threshold_pct = low_soc_threshold_pct

    # ------------------------------------------------------------------
    def estimate_vehicle(
        self,
        vehicle_id: str,
        telemetry_rows: List[sqlite3.Row],
        kwh_per_km: Optional[float] = None,
    ) -> RangeEstimate:
        """telemetry_rows: this vehicle's full history, ordered by cycle
        (ingestion.db.get_telemetry_for_vehicle's existing order) — only
        the LAST row is actually used; accepting the fuller history keeps
        this call shape symmetric with rul_model.py/charging_analyzer.py
        (both take full history too), so a future per-vehicle kwh_per_km
        derived from real distance data can reuse this signature as-is.
        Also accepts plain dicts (not just sqlite3.Row), since callers
        that already have a pandas DataFrame in hand (e.g.
        dashboard/components/health_chart.py, which already fetched
        telemetry for its own charts) can pass
        df.to_dict(orient="records") instead of re-querying the DB."""
        rate = kwh_per_km if kwh_per_km is not None else self.kwh_per_km

        if not telemetry_rows:
            return RangeEstimate(
                vehicle_id=vehicle_id, latest_cycle=None, soc_pct=None,
                capacity_kwh_remaining=None, kwh_per_km=rate,
                estimated_range_km=None, at_risk_of_stranding=False,
            )

        latest = telemetry_rows[-1]
        soc_pct = float(latest["soc_pct"])
        capacity_kwh_remaining = round(float(latest["capacity_kwh"]) * soc_pct / 100.0, 3)
        estimated_range_km = round(capacity_kwh_remaining / rate, 1) if rate > 0 else None

        at_risk = soc_pct <= self.low_soc_threshold_pct or (
            estimated_range_km is not None and estimated_range_km <= self.low_range_threshold_km
        )

        return RangeEstimate(
            vehicle_id=vehicle_id,
            latest_cycle=int(latest["cycle"]),
            soc_pct=soc_pct,
            capacity_kwh_remaining=capacity_kwh_remaining,
            kwh_per_km=rate,
            estimated_range_km=estimated_range_km,
            at_risk_of_stranding=bool(at_risk),
        )

    # ------------------------------------------------------------------
    def estimate_fleet(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids, get_telemetry_for_vehicle

        rows = []
        for vid in get_all_vehicle_ids(conn):
            telemetry_rows = get_telemetry_for_vehicle(conn, vid)
            rows.append(self.estimate_vehicle(vid, telemetry_rows).to_dict())
        return pd.DataFrame(rows, columns=[
            "vehicle_id", "latest_cycle", "soc_pct", "capacity_kwh_remaining",
            "kwh_per_km", "estimated_range_km", "at_risk_of_stranding",
        ])


if __name__ == "__main__":
    # Standalone sanity check, same pattern as rul_model.py/charging_analyzer.py.
    import os

    from ingestion.db import get_connection, init_db, insert_telemetry_batch
    from ingestion.schemas import TelemetryReading
    from simulator.config import SimulatorConfig
    from simulator.telemetry_generator import TelemetryGenerator

    cfg = SimulatorConfig(fleet_size=8, num_cycles=60, random_seed=21)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()

    test_db = os.path.join("data", "range_estimator_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    insert_telemetry_batch(conn, readings)

    estimator = RangeEstimator(
        kwh_per_km=cfg.avg_kwh_per_km,
        low_range_threshold_km=cfg.low_range_threshold_km,
        low_soc_threshold_pct=cfg.low_soc_threshold_pct,
    )
    result_df = estimator.estimate_fleet(conn)
    print(result_df.to_string(index=False))
    print(f"\n{int(result_df['at_risk_of_stranding'].sum())} vehicle(s) at risk of stranding")

    conn.close()
    os.remove(test_db)