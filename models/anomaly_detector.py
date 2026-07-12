"""
models/anomaly_detector.py

Two independent anomaly layers, per the project doc's §7.1/§7.2 split:

  1. Command/security anomalies — flags BMS commands showing signs of
     unauthorized use (Tirri Challenge mechanics): no matching
     maintenance ticket, GPS inconsistent with any known depot, or a
     frequency spike for that vehicle.

  2. Thermal anomalies — flagged purely from temperature telemetry,
     independent of command data.

This module only flags and scores; it does NOT decide what action to
take. That's the agent decision layer's job (Phase 4) — keeping
detection and decision separate is what makes the reasoning auditable.

Rule-based logic is the primary detector, exactly as specified in
§5.1 ("Rule-based logic, optionally supplemented with... IsolationForest").
IsolationForest is included as an optional secondary signal, not a
replacement for the explainable rules.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Default depot locations — mirrors simulator/config.py's defaults so the
# detector works out of the box against simulated data, but is a plain
# parameter (not an import) so models/ stays decoupled from simulator/.
DEFAULT_DEPOT_LOCATIONS: Tuple[Tuple[float, float], ...] = (
    (12.9716, 77.5946),   # Bengaluru
    (28.7041, 77.1025),   # Delhi
    (19.0760, 72.8777),   # Mumbai
)

DEFAULT_GPS_MISMATCH_KM = 2.0          # beyond this from every depot = "in motion on a public road"
DEFAULT_FREQUENCY_WINDOW_MINUTES = 10  # trailing window for burst detection
DEFAULT_FREQUENCY_THRESHOLD = 3        # >= this many commands in the window = spike

DEFAULT_SAFE_TEMP_C = 45.0
DEFAULT_CRITICAL_TEMP_C = 55.0
DEFAULT_ABNORMAL_RAMP_C_PER_MIN = 2.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _distance_to_nearest_depot(lat: float, lon: float, depots: Tuple[Tuple[float, float], ...]) -> float:
    return min(_haversine_km(lat, lon, dlat, dlon) for dlat, dlon in depots)


class AnomalyDetector:
    def __init__(
        self,
        depot_locations: Tuple[Tuple[float, float], ...] = DEFAULT_DEPOT_LOCATIONS,
        gps_mismatch_km: float = DEFAULT_GPS_MISMATCH_KM,
        frequency_window_minutes: int = DEFAULT_FREQUENCY_WINDOW_MINUTES,
        frequency_threshold: int = DEFAULT_FREQUENCY_THRESHOLD,
        safe_temp_c: float = DEFAULT_SAFE_TEMP_C,
        critical_temp_c: float = DEFAULT_CRITICAL_TEMP_C,
        abnormal_ramp_c_per_min: float = DEFAULT_ABNORMAL_RAMP_C_PER_MIN,
    ):
        self.depot_locations = depot_locations
        self.gps_mismatch_km = gps_mismatch_km
        self.frequency_window_minutes = frequency_window_minutes
        self.frequency_threshold = frequency_threshold
        self.safe_temp_c = safe_temp_c
        self.critical_temp_c = critical_temp_c
        self.abnormal_ramp_c_per_min = abnormal_ramp_c_per_min

    # ------------------------------------------------------------------
    # Command / security anomalies
    # ------------------------------------------------------------------
    def detect_command_anomalies(self, commands_df: pd.DataFrame) -> pd.DataFrame:
        """commands_df: rows from ingestion.db.get_commands_for_vehicle /
        get_all commands, with at least vehicle_id, timestamp, latitude,
        longitude, ticket_id. Returns the same rows with anomaly flags added.
        NOTE: is_attack (if present) is ground truth from the simulator and
        is deliberately never read here — only used afterwards to validate."""
        if commands_df.empty:
            return commands_df.assign(
                no_ticket_flag=pd.Series(dtype=bool),
                gps_mismatch_flag=pd.Series(dtype=bool),
                frequency_spike_flag=pd.Series(dtype=bool),
                signal_count=pd.Series(dtype=int),
                severity=pd.Series(dtype=str),
            )

        df = commands_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        # format="ISO8601": this column can contain a MIX of naive strings
        # (simulator-seeded rows, e.g. "2026-01-03T02:25:23.626357") and
        # tz-aware strings (live attack_trigger.py injections, e.g.
        # "...+00:00") once the dashboard's "Simulate Attack" button has been
        # used against a seeded DB. pandas' default to_datetime() infers a
        # single format from the first rows and raises ValueError the moment a
        # row does not match it; ISO8601 mode parses each value independently
        # as long as it is valid ISO 8601, which every timestamp this system
        # ever writes always is.
        # utc=True: pandas additionally refuses to represent a mix of naive
        # and tz-aware values in one column at all ("Mixed timezones
        # detected...") unless told how to reconcile them -- utc=True converts
        # every value to a common UTC representation first. The tz_localize(None)
        # call right below then strips that down to naive-but-UTC-normalized,
        # which is what every comparison later in this function assumes.
        # Real (non-simulated) clients may send timezone-aware timestamps
        # (e.g. attack_trigger.py's live injections use datetime.now(timezone.utc)).
        # pandas' Series.values silently drops tz info on tz-aware columns, so the
        # per-row pd.Timestamp(...) reconstruction in the frequency-spike loop below
        # would otherwise come back naive and raise "Invalid comparison" against this
        # still-tz-aware column. Normalize once, here, so every downstream comparison
        # in this function operates on consistently naive timestamps regardless of
        # whether the source was the simulator (already naive) or a live/real client.
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        # Sort for windowed frequency calc, but deliberately keep the original
        # index intact (no reset_index) so any index-based join a caller does
        # afterwards — e.g. reattaching a ground-truth column — stays aligned
        # to the correct row instead of silently pairing by position.
        df = df.sort_values(["vehicle_id", "timestamp"])

        # Signal 1: no matching maintenance ticket
        df["no_ticket_flag"] = df["ticket_id"].isnull()

        # Signal 2: GPS inconsistent with any known depot
        df["gps_mismatch_flag"] = df.apply(
            lambda row: _distance_to_nearest_depot(
                row["latitude"], row["longitude"], self.depot_locations
            ) > self.gps_mismatch_km,
            axis=1,
        )

        # Signal 3: frequency spike — count of commands for that vehicle in the
        # trailing window (including the command itself)
        window = pd.Timedelta(minutes=self.frequency_window_minutes)
        spike_flags = []
        for vid, group in df.groupby("vehicle_id"):
            ts = group["timestamp"].values
            counts = []
            for t in ts:
                t = pd.Timestamp(t)
                count = int(((group["timestamp"] > t - window) & (group["timestamp"] <= t)).sum())
                counts.append(count >= self.frequency_threshold)
            spike_flags.extend(zip(group.index, counts))
        spike_map = dict(spike_flags)
        df["frequency_spike_flag"] = df.index.map(spike_map)

        # Combined, explainable scoring — no hidden weighting. The agent
        # layer (Phase 4) decides what to do with this; this just counts
        # how many independent signals fired.
        df["signal_count"] = (
            df["no_ticket_flag"].astype(int)
            + df["gps_mismatch_flag"].astype(int)
            + df["frequency_spike_flag"].astype(int)
        )
        df["severity"] = df["signal_count"].map({0: "none", 1: "low", 2: "medium", 3: "high"})

        return df

    # ------------------------------------------------------------------
    # Thermal anomalies — independent of command data
    # ------------------------------------------------------------------
    def detect_thermal_anomalies(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """telemetry_df: rows with at least vehicle_id, cycle, timestamp,
        temperature_c. Returns the same rows with thermal flags added.
        NOTE: thermal_event_flag (if present) is simulator ground truth and
        is deliberately never read here — only used afterwards to validate."""
        if telemetry_df.empty:
            return telemetry_df.assign(
                sustained_high_temp_flag=pd.Series(dtype=bool),
                critical_temp_flag=pd.Series(dtype=bool),
                abnormal_ramp_flag=pd.Series(dtype=bool),
                thermal_anomaly=pd.Series(dtype=bool),
            )

        df = telemetry_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        # format="ISO8601": this column can contain a MIX of naive strings
        # (simulator-seeded rows, e.g. "2026-01-03T02:25:23.626357") and
        # tz-aware strings (live attack_trigger.py injections, e.g.
        # "...+00:00") once the dashboard's "Simulate Attack" button has been
        # used against a seeded DB. pandas' default to_datetime() infers a
        # single format from the first rows and raises ValueError the moment a
        # row does not match it; ISO8601 mode parses each value independently
        # as long as it is valid ISO 8601, which every timestamp this system
        # ever writes always is.
        # utc=True: pandas additionally refuses to represent a mix of naive
        # and tz-aware values in one column at all ("Mixed timezones
        # detected...") unless told how to reconcile them -- utc=True converts
        # every value to a common UTC representation first. The tz_localize(None)
        # call right below then strips that down to naive-but-UTC-normalized,
        # which is what every comparison later in this function assumes.
        # Same normalization as detect_command_anomalies, applied here defensively —
        # the ramp calc below only ever diffs two values pulled through the same
        # .values -> pd.Timestamp path so it happens not to be broken by tz-aware
        # input today, but keeping this column naive is what keeps it that way if
        # this function ever compares df["timestamp"] against an external Timestamp.
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        # Same reasoning as detect_command_anomalies: keep original index intact.
        df = df.sort_values(["vehicle_id", "cycle"])

        df["sustained_high_temp_flag"] = df["temperature_c"] >= self.safe_temp_c
        df["critical_temp_flag"] = df["temperature_c"] >= self.critical_temp_c

        ramp_flags = []
        for vid, group in df.groupby("vehicle_id"):
            temps = group["temperature_c"].values
            times = group["timestamp"].values
            flags = [False]  # first reading has no prior point to compare against
            for i in range(1, len(temps)):
                dt_minutes = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 60.0
                if dt_minutes <= 0:
                    flags.append(False)
                    continue
                rate = abs(temps[i] - temps[i - 1]) / dt_minutes
                flags.append(rate >= self.abnormal_ramp_c_per_min)
            ramp_flags.extend(zip(group.index, flags))
        ramp_map = dict(ramp_flags)
        df["abnormal_ramp_flag"] = df.index.map(ramp_map)

        df["thermal_anomaly"] = (
            df["sustained_high_temp_flag"] | df["critical_temp_flag"] | df["abnormal_ramp_flag"]
        )

        return df

    # ------------------------------------------------------------------
    # Optional supplemental signal — IsolationForest over telemetry features.
    # Secondary only: never the primary detector, per §5.1.
    # ------------------------------------------------------------------
    def fit_isolation_forest_scores(
        self, telemetry_df: pd.DataFrame,
        features: Tuple[str, ...] = ("temperature_c", "voltage", "soc_pct", "dod_pct"),
        contamination: float = 0.05,
    ) -> pd.DataFrame:
        from sklearn.ensemble import IsolationForest

        if telemetry_df.empty or len(telemetry_df) < 10:
            return telemetry_df.assign(isolation_forest_outlier=pd.Series(dtype=bool))

        df = telemetry_df.copy()
        X = df[list(features)].fillna(df[list(features)].mean())
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(X)  # -1 = outlier, 1 = inlier
        df["isolation_forest_outlier"] = preds == -1
        return df


if __name__ == "__main__":
    # Standalone sanity check: fresh simulator data, run both detectors,
    # and compare against simulator ground truth (is_attack / thermal_event_flag)
    # purely to sanity-check detector behaviour — never done in production logic.
    from simulator.config import SimulatorConfig
    from simulator.telemetry_generator import TelemetryGenerator
    from simulator.maintenance_generator import MaintenanceGenerator
    from simulator.attack_injector import AttackInjector

    cfg = SimulatorConfig(fleet_size=10, num_cycles=100, random_seed=55)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)
    mgen = MaintenanceGenerator(cfg)
    tickets_df = mgen.generate_fleet_tickets(bounds)
    ainj = AttackInjector(cfg)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    detector = AnomalyDetector()

    cmd_result = detector.detect_command_anomalies(commands_df.drop(columns=["is_attack"]))
    cmd_result["is_attack"] = commands_df["is_attack"]  # reattach ground truth for validation only

    tp = int(((cmd_result["signal_count"] >= 1) & (cmd_result["is_attack"] == True)).sum())
    fn = int(((cmd_result["signal_count"] == 0) & (cmd_result["is_attack"] == True)).sum())
    fp = int(((cmd_result["signal_count"] >= 1) & (cmd_result["is_attack"] == False)).sum())
    tn = int(((cmd_result["signal_count"] == 0) & (cmd_result["is_attack"] == False)).sum())
    print(f"Command anomaly detection vs ground truth (>=1 signal flagged as suspicious):")
    print(f"  TP={tp} FN={fn} FP={fp} TN={tn}")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    print(f"  recall={recall:.2f} precision={precision:.2f}")

    therm_result = detector.detect_thermal_anomalies(telem_df.drop(columns=["thermal_event_flag"]))
    therm_result["thermal_event_flag"] = telem_df["thermal_event_flag"]
    t_tp = int((therm_result["thermal_anomaly"] & therm_result["thermal_event_flag"]).sum())
    t_actual = int(therm_result["thermal_event_flag"].sum())
    t_flagged = int(therm_result["thermal_anomaly"].sum())
    print(f"\nThermal anomaly detection: {t_flagged} flagged, {t_actual} true events, {t_tp} overlap")