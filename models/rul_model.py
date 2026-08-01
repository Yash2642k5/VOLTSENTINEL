from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DEFAULT_END_OF_LIFE_CAPACITY_PCT = 70.0
MIN_POINTS_FOR_FIT = 5


def _exp_decay(cycle: np.ndarray, a: float, b: float) -> np.ndarray:
    #Capacity fade model: capacity_pct(cycle) = a * exp(-b * cycle).
    return a * np.exp(-b * cycle)


@dataclass
class RULResult:
    vehicle_id: str
    status: str                       # "healthy" | "watch" | "degraded" | "critical" | "insufficient_data" | "no_fit"
    current_cycle: Optional[int] = None
    current_capacity_pct: Optional[float] = None
    fitted_a: Optional[float] = None      # fitted initial capacity_pct (~100 if healthy sensor)
    fitted_decay_rate: Optional[float] = None  # b in a * exp(-b * cycle)
    r_squared: Optional[float] = None
    eol_cycle: Optional[float] = None     # projected cycle at which EOL threshold is crossed
    rul_cycles: Optional[float] = None    # cycles remaining until EOL, floored at 0
    end_of_life_capacity_pct: float = DEFAULT_END_OF_LIFE_CAPACITY_PCT

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RULModel:
    def __init__(self, end_of_life_capacity_pct: float = DEFAULT_END_OF_LIFE_CAPACITY_PCT):
        self.end_of_life_capacity_pct = end_of_life_capacity_pct

    # Status banding — kept as simple, explainable thresholds on current capacity relative to end-of-life, not a hidden model output.
    def _status_from_capacity(self, current_capacity_pct: float) -> str:
        eol = self.end_of_life_capacity_pct
        margin = 100.0 - eol
        if current_capacity_pct <= eol:
            return "critical"
        elif current_capacity_pct <= eol + margin * 0.33:   # within ~10pt of EOL
            return "degraded"
        elif current_capacity_pct <= eol + margin * 0.66:   # within ~20pt of EOL
            return "watch"
        return "healthy"

    # Core fit — single vehicle
    def fit_vehicle(self, vehicle_id: str, telemetry_rows: List[sqlite3.Row]) -> RULResult:
        if len(telemetry_rows) < MIN_POINTS_FOR_FIT:
            return RULResult(
                vehicle_id=vehicle_id,
                status="insufficient_data",
                current_cycle=telemetry_rows[-1]["cycle"] if telemetry_rows else None,
                current_capacity_pct=telemetry_rows[-1]["capacity_pct_of_rated"] if telemetry_rows else None,
                end_of_life_capacity_pct=self.end_of_life_capacity_pct,
            )

        cycles = np.array([r["cycle"] for r in telemetry_rows], dtype=float)
        capacity_pct = np.array([r["capacity_pct_of_rated"] for r in telemetry_rows], dtype=float)
        current_cycle = int(cycles[-1])
        current_capacity_pct = float(capacity_pct[-1])

        try:
            # Initial guess: a ~ first reading, b ~ small positive decay rate.
            p0 = (float(capacity_pct[0]) if capacity_pct[0] > 0 else 100.0, 0.001)
            popt, _ = curve_fit(
                _exp_decay, cycles, capacity_pct, p0=p0,
                bounds=([1.0, 0.0], [200.0, 1.0]), maxfev=5000,
            )
            a, b = popt

            predicted = _exp_decay(cycles, a, b)
            ss_res = float(np.sum((capacity_pct - predicted) ** 2))
            ss_tot = float(np.sum((capacity_pct - capacity_pct.mean()) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        except (RuntimeError, ValueError):
            return RULResult(
                vehicle_id=vehicle_id,
                status="no_fit",
                current_cycle=current_cycle,
                current_capacity_pct=current_capacity_pct,
                end_of_life_capacity_pct=self.end_of_life_capacity_pct,
            )

        eol_cycle: Optional[float]
        rul_cycles: Optional[float]
        if b <= 1e-6 or a <= self.end_of_life_capacity_pct:
            eol_cycle = None
            rul_cycles = None
        else:
            eol_cycle = float(np.log(a / self.end_of_life_capacity_pct) / b)
            rul_cycles = max(eol_cycle - current_cycle, 0.0)

        return RULResult(
            vehicle_id=vehicle_id,
            status=self._status_from_capacity(current_capacity_pct),
            current_cycle=current_cycle,
            current_capacity_pct=round(current_capacity_pct, 2),
            fitted_a=round(float(a), 3),
            fitted_decay_rate=round(float(b), 6),
            r_squared=round(float(r_squared), 4),
            eol_cycle=round(eol_cycle, 1) if eol_cycle is not None else None,
            rul_cycles=round(rul_cycles, 1) if rul_cycles is not None else None,
            end_of_life_capacity_pct=self.end_of_life_capacity_pct,
        )

    def fit_fleet(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids, get_telemetry_for_vehicle

        results = []
        for vehicle_id in get_all_vehicle_ids(conn):
            rows = get_telemetry_for_vehicle(conn, vehicle_id)
            results.append(self.fit_vehicle(vehicle_id, rows).to_dict())

        return pd.DataFrame(results)


if __name__ == "__main__":
    import os

    from ingestion.db import (
        get_connection, init_db, insert_telemetry_batch,
        insert_maintenance_batch, insert_command_batch,
    )
    from ingestion.schemas import TelemetryReading, MaintenanceTicket, CommandEvent
    from simulator.config import SimulatorConfig
    from simulator.telemetry_generator import TelemetryGenerator
    from simulator.maintenance_generator import MaintenanceGenerator
    from simulator.attack_injector import AttackInjector

    cfg = SimulatorConfig(fleet_size=6, num_cycles=300, random_seed=11)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    test_db = os.path.join("data", "rul_model_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    import math
    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    tickets = [MaintenanceTicket(**r) for r in tickets_df.to_dict(orient="records")]
    commands = []
    for r in commands_df.to_dict(orient="records"):
        if isinstance(r.get("ticket_id"), float) and math.isnan(r["ticket_id"]):
            r["ticket_id"] = None
        commands.append(CommandEvent(**r))

    insert_telemetry_batch(conn, readings)
    insert_maintenance_batch(conn, tickets)
    insert_command_batch(conn, commands)

    model = RULModel(end_of_life_capacity_pct=cfg.end_of_life_capacity_pct * 100)
    results_df = model.fit_fleet(conn)

    print(results_df[[
        "vehicle_id", "status", "current_cycle", "current_capacity_pct",
        "fitted_decay_rate", "r_squared", "rul_cycles",
    ]].to_string(index=False))

    print(f"\nMean R^2 across fleet: {results_df['r_squared'].mean():.4f}")
    conn.close()
    os.remove(test_db)