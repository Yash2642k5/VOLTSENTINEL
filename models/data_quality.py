"""Data-quality checks — missing-cycle gaps, stale-sensor flags, and
out-of-range sensor values. Feeds risk_engine.py's fleet profile as an
independent risk signal, alongside RUL/thermal/security/charging."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_STALE_HOURS = 48.0
# capacity shouldn't meaningfully increase cycle-over-cycle; set well above the
# simulator's own capacity_noise_std_pct so routine noise doesn't trip this check
DEFAULT_CAPACITY_INCREASE_JUMP_PCT = 8.0
DEFAULT_TEMPERATURE_RANGE_C: Tuple[float, float] = (-20.0, 120.0)
DEFAULT_VOLTAGE_RANGE_V: Tuple[float, float] = (0.0, 1000.0)
DEFAULT_SOC_RANGE_PCT: Tuple[float, float] = (0.0, 100.0)


@dataclass
class DataQualityProfile:
    vehicle_id: str
    missing_cycle_count: int
    is_stale: bool
    hours_since_last_reading: Optional[float]
    out_of_range_jump_count: int
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["issues"] = ",".join(self.issues)
        return d


class DataQualityAnalyzer:
    def __init__(
        self,
        stale_hours: float = DEFAULT_STALE_HOURS,
        capacity_increase_jump_pct: float = DEFAULT_CAPACITY_INCREASE_JUMP_PCT,
        temperature_range_c: Tuple[float, float] = DEFAULT_TEMPERATURE_RANGE_C,
        voltage_range_v: Tuple[float, float] = DEFAULT_VOLTAGE_RANGE_V,
        soc_range_pct: Tuple[float, float] = DEFAULT_SOC_RANGE_PCT,
    ):
        self.stale_hours = stale_hours
        self.capacity_increase_jump_pct = capacity_increase_jump_pct
        self.temperature_range_c = temperature_range_c
        self.voltage_range_v = voltage_range_v
        self.soc_range_pct = soc_range_pct

    def analyze_vehicle(
        self, conn: sqlite3.Connection, vehicle_id: str, now: Optional[datetime] = None,
    ) -> DataQualityProfile:
        from ingestion.db import get_telemetry_for_vehicle

        now = now or datetime.now(timezone.utc)
        rows = get_telemetry_for_vehicle(conn, vehicle_id)
        if not rows:
            return DataQualityProfile(
                vehicle_id=vehicle_id, missing_cycle_count=0, is_stale=True,
                hours_since_last_reading=None, out_of_range_jump_count=0,
                issues=["no_telemetry"],
            )

        df = pd.DataFrame([dict(r) for r in rows]).sort_values("cycle")
        issues: List[str] = []

        cycles = df["cycle"].tolist()
        expected = cycles[-1] - cycles[0] + 1
        missing_cycle_count = max(0, expected - len(cycles))
        if missing_cycle_count > 0:
            issues.append("missing_cycles")

        timestamps = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        now_ts = pd.Timestamp(now) if now.tzinfo else pd.Timestamp(now, tz="UTC")
        hours_since_last_reading = round((now_ts - timestamps.iloc[-1]).total_seconds() / 3600, 1)
        is_stale = hours_since_last_reading > self.stale_hours
        if is_stale:
            issues.append("stale_sensor")

        out_of_range_jump_count = (
            int((df["capacity_pct_of_rated"].diff() > self.capacity_increase_jump_pct).sum())
            + int((~df["temperature_c"].between(*self.temperature_range_c)).sum())
            + int((~df["voltage"].between(*self.voltage_range_v)).sum())
            + int((~df["soc_pct"].between(*self.soc_range_pct)).sum())
        )
        if out_of_range_jump_count > 0:
            issues.append("out_of_range_values")

        return DataQualityProfile(
            vehicle_id=vehicle_id,
            missing_cycle_count=missing_cycle_count,
            is_stale=is_stale,
            hours_since_last_reading=hours_since_last_reading,
            out_of_range_jump_count=out_of_range_jump_count,
            issues=issues,
        )

    def analyze_fleet(self, conn: sqlite3.Connection, now: Optional[datetime] = None) -> pd.DataFrame:
        from ingestion.db import get_all_vehicle_ids

        vehicle_ids = get_all_vehicle_ids(conn)
        if not vehicle_ids:
            return pd.DataFrame(columns=[
                "vehicle_id", "missing_cycle_count", "is_stale",
                "hours_since_last_reading", "out_of_range_jump_count", "issues",
            ])
        rows = [self.analyze_vehicle(conn, vid, now=now).to_dict() for vid in vehicle_ids]
        return pd.DataFrame(rows)
