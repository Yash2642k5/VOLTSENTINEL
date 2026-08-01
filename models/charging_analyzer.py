from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

DEFAULT_HIGH_DOD_THRESHOLD_PCT = 85.0
DEFAULT_FLEET_FAST_CHARGE_BASELINE_PCT = 25.0
DEFAULT_TREND_WINDOW = 20  # number of most-recent cycles considered "recent" vs "early"


@dataclass
class ChargingProfile:
    vehicle_id: str
    total_cycles: int
    fast_charge_frequency_pct: float
    mean_dod_pct: float
    high_dod_frequency_pct: float
    fleet_fast_charge_baseline_pct: float
    fast_charge_vs_baseline_pct: float   # positive = charges fast more often than fleet average
    stress_trend: str                    # "increasing" | "stable" | "decreasing" | "insufficient_data"
    charge_stress_score: float           # 0-100, higher = more stressful charging behaviour
    suggested_policy: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DriverChargingProfile:
    driver_id: str
    vehicle_count: int
    total_cycles: int
    fast_charge_frequency_pct: float
    mean_dod_pct: float
    high_dod_frequency_pct: float
    fleet_fast_charge_baseline_pct: float
    fast_charge_vs_baseline_pct: float
    stress_trend: str
    charge_stress_score: float
    suggested_policy: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ChargingAnalyzer:
    def __init__(
        self,
        high_dod_threshold_pct: float = DEFAULT_HIGH_DOD_THRESHOLD_PCT,
        trend_window: int = DEFAULT_TREND_WINDOW,
    ):
        self.high_dod_threshold_pct = high_dod_threshold_pct
        self.trend_window = trend_window

    # ------------------------------------------------------------------
    def _stress_trend(self, is_fast_charge: pd.Series, dod_pct: pd.Series) -> str:
        n = len(dod_pct)
        w = self.trend_window
        if n < w * 2:
            return "insufficient_data"

        early_stress = is_fast_charge.iloc[:w].mean() * 50 + (dod_pct.iloc[:w].mean() / 100) * 50
        recent_stress = is_fast_charge.iloc[-w:].mean() * 50 + (dod_pct.iloc[-w:].mean() / 100) * 50

        delta = recent_stress - early_stress
        if delta > 5:
            return "increasing"
        elif delta < -5:
            return "decreasing"
        return "stable"

    def _suggest_policy(self, profile_partial: dict) -> Optional[str]:
        suggestions = []
        if profile_partial["fast_charge_vs_baseline_pct"] > 15:
            suggestions.append("cap fast-charge frequency")
        if profile_partial["mean_dod_pct"] >= self.high_dod_threshold_pct:
            suggestions.append(f"limit depth-of-discharge below {self.high_dod_threshold_pct:.0f}%")
        if profile_partial["stress_trend"] == "increasing":
            suggestions.append("flag for driver coaching — stress trend worsening")

        return "; ".join(suggestions) if suggestions else None

    # ------------------------------------------------------------------
    def _compute_charging_stats(
        self, telemetry_rows: List[sqlite3.Row], fleet_baseline_pct: float
    ) -> dict:
        if not telemetry_rows:
            return {
                "total_cycles": 0, "fast_charge_frequency_pct": 0.0,
                "mean_dod_pct": 0.0, "high_dod_frequency_pct": 0.0,
                "fleet_fast_charge_baseline_pct": round(fleet_baseline_pct, 2),
                "fast_charge_vs_baseline_pct": 0.0, "stress_trend": "insufficient_data",
                "charge_stress_score": 0.0, "suggested_policy": None,
            }

        is_fast_charge = pd.Series([bool(r["is_fast_charge"]) for r in telemetry_rows])
        dod_pct = pd.Series([float(r["dod_pct"]) for r in telemetry_rows])

        fast_charge_frequency_pct = round(float(is_fast_charge.mean() * 100), 2)
        mean_dod_pct = round(float(dod_pct.mean()), 2)
        high_dod_frequency_pct = round(float((dod_pct >= self.high_dod_threshold_pct).mean() * 100), 2)
        stress_trend = self._stress_trend(is_fast_charge, dod_pct)

        # Composite 0-100 stress score — simple weighted average of three
        charge_stress_score = round(
            0.4 * fast_charge_frequency_pct
            + 0.4 * min(mean_dod_pct, 100.0)
            + 0.2 * high_dod_frequency_pct,
            2,
        )

        profile_partial = {
            "fast_charge_vs_baseline_pct": round(fast_charge_frequency_pct - fleet_baseline_pct, 2),
            "mean_dod_pct": mean_dod_pct,
            "stress_trend": stress_trend,
        }

        return {
            "total_cycles": len(telemetry_rows),
            "fast_charge_frequency_pct": fast_charge_frequency_pct,
            "mean_dod_pct": mean_dod_pct,
            "high_dod_frequency_pct": high_dod_frequency_pct,
            "fleet_fast_charge_baseline_pct": round(fleet_baseline_pct, 2),
            "fast_charge_vs_baseline_pct": profile_partial["fast_charge_vs_baseline_pct"],
            "stress_trend": stress_trend,
            "charge_stress_score": charge_stress_score,
            "suggested_policy": self._suggest_policy(profile_partial),
        }

    def analyze_vehicle(
        self, vehicle_id: str, telemetry_rows: List[sqlite3.Row], fleet_baseline_pct: float
    ) -> ChargingProfile:
        stats = self._compute_charging_stats(telemetry_rows, fleet_baseline_pct)
        return ChargingProfile(vehicle_id=vehicle_id, **stats)

    def analyze_fleet(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids, get_telemetry_for_vehicle

        vehicle_ids = get_all_vehicle_ids(conn)
        all_rows = {vid: get_telemetry_for_vehicle(conn, vid) for vid in vehicle_ids}

        all_fast_charge = [
            bool(r["is_fast_charge"]) for rows in all_rows.values() for r in rows
        ]
        fleet_baseline_pct = (
            round(sum(all_fast_charge) / len(all_fast_charge) * 100, 2)
            if all_fast_charge else DEFAULT_FLEET_FAST_CHARGE_BASELINE_PCT
        )

        results = [
            self.analyze_vehicle(vid, rows, fleet_baseline_pct).to_dict()
            for vid, rows in all_rows.items()
        ]
        return pd.DataFrame(results)

    #Driver-Level Coaching
    def analyze_driver(
        self,
        driver_id: str,
        telemetry_rows: List[sqlite3.Row],
        fleet_baseline_pct: float,
        vehicle_count: int = 0,
    ) -> DriverChargingProfile:
        stats = self._compute_charging_stats(telemetry_rows, fleet_baseline_pct)
        return DriverChargingProfile(driver_id=driver_id, vehicle_count=vehicle_count, **stats)

    def analyze_fleet_drivers(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import (
            get_all_drivers, get_all_vehicle_ids, get_assignments_for_driver,
            get_telemetry_for_vehicle,
        )

        vehicle_ids = get_all_vehicle_ids(conn)

        telemetry_cache = {vid: get_telemetry_for_vehicle(conn, vid) for vid in vehicle_ids}

        all_fast_charge = [
            bool(r["is_fast_charge"]) for rows in telemetry_cache.values() for r in rows
        ]
        fleet_baseline_pct = (
            round(sum(all_fast_charge) / len(all_fast_charge) * 100, 2)
            if all_fast_charge else DEFAULT_FLEET_FAST_CHARGE_BASELINE_PCT
        )

        results = []
        for driver in get_all_drivers(conn):
            driver_id = driver["driver_id"]
            assignments = get_assignments_for_driver(conn, driver_id)

            driver_rows: List[sqlite3.Row] = []
            vehicles_driven = set()
            for assignment in assignments:
                vid = assignment["vehicle_id"]
                vehicles_driven.add(vid)
                start = pd.Timestamp(assignment["shift_start"])
                end = pd.Timestamp(assignment["shift_end"]) if assignment["shift_end"] else None
                for row in telemetry_cache.get(vid, []):
                    ts = pd.Timestamp(row["timestamp"])
                    if ts >= start and (end is None or ts <= end):
                        driver_rows.append(row)

            # Chronological order matters for _stress_trend's early-vs-recent
            # window comparison — rows arrive vehicle-by-vehicle above, not
            # already time-ordered across vehicles.
            driver_rows.sort(key=lambda r: r["timestamp"])

            profile = self.analyze_driver(
                driver_id, driver_rows, fleet_baseline_pct, vehicle_count=len(vehicles_driven)
            )
            results.append(profile.to_dict())

        return pd.DataFrame(results, columns=[
            "driver_id", "vehicle_count", "total_cycles", "fast_charge_frequency_pct",
            "mean_dod_pct", "high_dod_frequency_pct", "fleet_fast_charge_baseline_pct",
            "fast_charge_vs_baseline_pct", "stress_trend", "charge_stress_score",
            "suggested_policy",
        ])


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

    cfg = SimulatorConfig(fleet_size=8, num_cycles=80, random_seed=33)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    test_db = os.path.join("data", "charging_analyzer_test.db")
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

    analyzer = ChargingAnalyzer()
    profiles_df = analyzer.analyze_fleet(conn)
    print(profiles_df[[
        "vehicle_id", "fast_charge_frequency_pct", "mean_dod_pct",
        "fast_charge_vs_baseline_pct", "stress_trend", "charge_stress_score", "suggested_policy",
    ]].to_string(index=False))

    conn.close()
    os.remove(test_db)