"""
models/risk_engine.py

Merges the three parallel analytical layers — RUL (rul_model.py),
command/thermal anomalies (anomaly_detector.py), and charging patterns
(charging_analyzer.py) — into a single per-asset risk profile.

This is the "converge into a merged per-asset risk profile" step from
§5 of the project doc, and it's the direct input to the agent decision
layer's "Perceive" step in Phase 4 (agent/decision_engine.py).

Important: this module aggregates and summarizes signals — it does
NOT decide what action to take. The `overall_risk_level` computed here
is a simple, transparent banding meant to help a human skim the fleet
view; the agent layer is what actually reasons over the full profile
and emits maintenance triggers, charge policies, or escalations. Kept
separate on purpose, matching the doc's Perceive -> Reason -> Decide -> Act
split (§6.1).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from .anomaly_detector import AnomalyDetector
from .charging_analyzer import ChargingAnalyzer
from .rul_model import RULModel


class RiskEngine:
    def __init__(
        self,
        rul_model: Optional[RULModel] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        charging_analyzer: Optional[ChargingAnalyzer] = None,
    ):
        self.rul_model = rul_model or RULModel()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.charging_analyzer = charging_analyzer or ChargingAnalyzer()

    # ------------------------------------------------------------------
    def _aggregate_thermal(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids, get_telemetry_for_vehicle

        rows = []
        for vid in get_all_vehicle_ids(conn):
            telemetry_rows = get_telemetry_for_vehicle(conn, vid)
            telem_df = pd.DataFrame([dict(r) for r in telemetry_rows])
            if telem_df.empty:
                rows.append({"vehicle_id": vid, "thermal_anomaly_count": 0,
                             "critical_temp_count": 0, "latest_temp_c": None})
                continue

            result = self.anomaly_detector.detect_thermal_anomalies(telem_df)
            rows.append({
                "vehicle_id": vid,
                "thermal_anomaly_count": int(result["thermal_anomaly"].sum()),
                "critical_temp_count": int(result["critical_temp_flag"].sum()),
                "latest_temp_c": float(result.sort_values("cycle").iloc[-1]["temperature_c"]),
            })
        return pd.DataFrame(rows)

    def _aggregate_security(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids, get_commands_for_vehicle

        rows = []
        for vid in get_all_vehicle_ids(conn):
            command_rows = get_commands_for_vehicle(conn, vid)
            cmd_df = pd.DataFrame([dict(r) for r in command_rows])
            if cmd_df.empty:
                rows.append({
                    "vehicle_id": vid, "unticketed_command_count": 0,
                    "max_security_severity": "none", "suspicious_command_count": 0,
                })
                continue

            result = self.anomaly_detector.detect_command_anomalies(cmd_df)
            severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
            max_severity = max(result["severity"], key=lambda s: severity_rank[s])
            rows.append({
                "vehicle_id": vid,
                "unticketed_command_count": int(result["no_ticket_flag"].sum()),
                "max_security_severity": max_severity,
                "suspicious_command_count": int((result["signal_count"] >= 1).sum()),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _overall_risk_level(self, row: pd.Series) -> str:
        """Simple, transparent banding for fleet-view skimming — counts how
        many domains show a concerning signal. NOT the agent's decision;
        the agent (Phase 4) reasons over the full profile independently."""
        concerns = 0
        if row.get("status") in ("degraded", "critical"):
            concerns += 1
        if row.get("thermal_anomaly_count", 0) > 0:
            concerns += 1
        if row.get("max_security_severity") in ("medium", "high"):
            concerns += 1
        if row.get("charge_stress_score", 0) >= 60:
            concerns += 1

        if concerns >= 3:
            return "high"
        elif concerns == 2:
            return "medium"
        elif concerns == 1:
            return "low"
        return "minimal"

    # ------------------------------------------------------------------
    _EMPTY_PROFILE_COLUMNS = [
        "vehicle_id", "status", "current_cycle", "current_capacity_pct",
        "fitted_a", "fitted_decay_rate", "r_squared", "eol_cycle",
        "rul_cycles", "end_of_life_capacity_pct",
        "thermal_anomaly_count", "critical_temp_count", "latest_temp_c",
        "unticketed_command_count", "max_security_severity", "suspicious_command_count",
        "total_cycles", "fast_charge_frequency_pct", "mean_dod_pct",
        "high_dod_frequency_pct", "fleet_fast_charge_baseline_pct",
        "fast_charge_vs_baseline_pct", "stress_trend", "charge_stress_score",
        "suggested_policy", "overall_risk_level",
    ]

    def build_fleet_profile(self, conn: sqlite3.Connection) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids

        # An empty DB (nothing ingested yet, or freshly created) means every
        # per-model DataFrame below comes back as pd.DataFrame([]) — which has
        # NO columns at all, not even vehicle_id. Merging on "vehicle_id" in
        # that state raises KeyError instead of just producing an empty
        # profile. Short-circuit here so the dashboard's existing "No
        # vehicles in the fleet profile yet" empty-state message (already
        # handled downstream in fleet_map.py / utils.py) actually gets a
        # chance to render, instead of the whole page crashing first.
        if not get_all_vehicle_ids(conn):
            return pd.DataFrame(columns=self._EMPTY_PROFILE_COLUMNS)

        rul_df = self.rul_model.fit_fleet(conn)
        thermal_df = self._aggregate_thermal(conn)
        security_df = self._aggregate_security(conn)
        charging_df = self.charging_analyzer.analyze_fleet(conn)

        profile = (
            rul_df
            .merge(thermal_df, on="vehicle_id", how="outer")
            .merge(security_df, on="vehicle_id", how="outer")
            .merge(charging_df, on="vehicle_id", how="outer", suffixes=("", "_charging"))
        )

        profile["overall_risk_level"] = profile.apply(self._overall_risk_level, axis=1)
        return profile


if __name__ == "__main__":
    import math
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

    cfg = SimulatorConfig(fleet_size=10, num_cycles=250, random_seed=2026)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    test_db = os.path.join("data", "risk_engine_test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

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

    engine = RiskEngine(rul_model=RULModel(end_of_life_capacity_pct=cfg.end_of_life_capacity_pct * 100))
    profile_df = engine.build_fleet_profile(conn)

    cols = [
        "vehicle_id", "status", "rul_cycles", "thermal_anomaly_count",
        "max_security_severity", "suspicious_command_count",
        "charge_stress_score", "stress_trend", "overall_risk_level",
    ]
    print(profile_df[cols].to_string(index=False))

    print("\nRisk level distribution:")
    print(profile_df["overall_risk_level"].value_counts())

    conn.close()
    os.remove(test_db)