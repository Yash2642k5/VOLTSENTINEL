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

Future Roadmap Feature 8 — Weather-Aware Range Estimate:
  Battery efficiency (kWh/km) is materially affected by ambient
  temperature — worse in the cold (higher internal resistance,
  cabin/pack heating draw) and, to a smaller degree, in extreme heat
  (thermal-management/cooling load). estimate_vehicle() now accepts an
  optional `ambient_temp_c` and applies a plain, explainable multiplier
  on top of kwh_per_km via `_weather_adjustment_factor` — a correction
  term on the existing baseline, not a replacement model, matching
  rul_model.py's "transparent curve fit over a black-box model"
  philosophy. Live weather itself comes from models/weather_client.py
  (a separate, network-isolated, key-free client); this module never
  imports `requests` or talks to the network directly — only
  estimate_vehicle_live()/estimate_fleet() accept an already-constructed
  WeatherClient and treat a None reading (client omitted, lookup failed,
  vehicle has no known location) as "stay weather-agnostic," so every
  existing caller of estimate_vehicle()/estimate_fleet() without a
  weather_client behaves exactly as before this feature was added.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_KWH_PER_KM = 0.06                # e-rickshaw-class energy consumption
DEFAULT_LOW_RANGE_THRESHOLD_KM = 15.0    # below this = "at risk of stranding"
DEFAULT_LOW_SOC_THRESHOLD_PCT = 20.0     # belt-and-braces SoC-only check

# ----------------------------------------------------------------------
# Weather adjustment (Future Roadmap Feature 8) — a plain, explainable
# correction curve on top of kwh_per_km, not a black-box model, matching
# rul_model.py's "transparent curve fit over a hidden model" philosophy.
# EV/e-rickshaw efficiency is roughly optimal in a mild band; it worsens
# in the cold (higher internal resistance, cabin/pack heating draw) and,
# to a smaller degree, in extreme heat (thermal-management/cooling load
# — relevant here since simulator/config.py's own ambient_temp_mean_c is
# 32°C, i.e. this fleet mostly lives on the hot side of the curve).
# ----------------------------------------------------------------------
DEFAULT_COOL_REFERENCE_TEMP_C = 15.0         # below this, cold penalty kicks in
DEFAULT_COOL_PENALTY_PCT_PER_C = 1.5         # % extra kWh/km per °C below the cool reference
DEFAULT_WARM_REFERENCE_TEMP_C = 30.0         # above this, heat penalty kicks in
DEFAULT_WARM_PENALTY_PCT_PER_C = 0.8         # % extra kWh/km per °C above the warm reference
DEFAULT_MAX_WEATHER_ADJUSTMENT_FACTOR = 1.6  # sanity cap — never more than +60% from weather alone

# Depot locations — mirrors simulator/config.py's defaults so this works
# out of the box against simulated data, but stays a plain parameter, not
# an import, same reasoning as models/anomaly_detector.py's
# DEFAULT_DEPOT_LOCATIONS: models/ should work against whatever is in
# SQLite regardless of source. Used only to pick a "regional" coordinate
# to query weather for, since telemetry itself carries no per-row GPS —
# only commands do (a vehicle's last known location).
DEFAULT_DEPOT_LOCATIONS: Tuple[Tuple[float, float], ...] = (
    (12.9716, 77.5946),   # Bengaluru
    (28.7041, 77.1025),   # Delhi
    (19.0760, 72.8777),   # Mumbai
)


def _nearest_depot(
    lat: float, lon: float, depots: Tuple[Tuple[float, float], ...]
) -> Tuple[float, float]:
    """Nearest depot by plain squared distance — good enough for picking
    which region's weather to query; doesn't need haversine precision
    the way anomaly_detector.py's GPS-mismatch check does."""
    return min(depots, key=lambda d: (d[0] - lat) ** 2 + (d[1] - lon) ** 2)


@dataclass
class RangeEstimate:
    vehicle_id: str
    latest_cycle: Optional[int]
    soc_pct: Optional[float]
    capacity_kwh_remaining: Optional[float]
    kwh_per_km: float
    estimated_range_km: Optional[float]
    at_risk_of_stranding: bool
    ambient_temp_c: Optional[float] = None    # NEW (Feature 8) — live weather, if available
    weather_adjustment_factor: float = 1.0    # NEW (Feature 8) — 1.0 = no adjustment applied

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RangeEstimator:
    def __init__(
        self,
        kwh_per_km: float = DEFAULT_KWH_PER_KM,
        low_range_threshold_km: float = DEFAULT_LOW_RANGE_THRESHOLD_KM,
        low_soc_threshold_pct: float = DEFAULT_LOW_SOC_THRESHOLD_PCT,
        cool_reference_temp_c: float = DEFAULT_COOL_REFERENCE_TEMP_C,
        cool_penalty_pct_per_c: float = DEFAULT_COOL_PENALTY_PCT_PER_C,
        warm_reference_temp_c: float = DEFAULT_WARM_REFERENCE_TEMP_C,
        warm_penalty_pct_per_c: float = DEFAULT_WARM_PENALTY_PCT_PER_C,
        max_weather_adjustment_factor: float = DEFAULT_MAX_WEATHER_ADJUSTMENT_FACTOR,
    ):
        self.kwh_per_km = kwh_per_km
        self.low_range_threshold_km = low_range_threshold_km
        self.low_soc_threshold_pct = low_soc_threshold_pct
        self.cool_reference_temp_c = cool_reference_temp_c
        self.cool_penalty_pct_per_c = cool_penalty_pct_per_c
        self.warm_reference_temp_c = warm_reference_temp_c
        self.warm_penalty_pct_per_c = warm_penalty_pct_per_c
        self.max_weather_adjustment_factor = max_weather_adjustment_factor

    # ------------------------------------------------------------------
    # Weather adjustment (Future Roadmap Feature 8)
    # ------------------------------------------------------------------
    def _weather_adjustment_factor(self, temp_c: Optional[float]) -> float:
        """1.0 when temp_c is None (no live weather — stay weather-
        agnostic, exactly today's behaviour) or when temp_c falls inside
        the mild reference band. Otherwise a plain linear penalty per °C
        outside that band, capped at max_weather_adjustment_factor so a
        garbage/extreme API reading can't blow up the range estimate."""
        if temp_c is None:
            return 1.0

        factor = 1.0
        if temp_c < self.cool_reference_temp_c:
            factor += (self.cool_reference_temp_c - temp_c) * self.cool_penalty_pct_per_c / 100.0
        elif temp_c > self.warm_reference_temp_c:
            factor += (temp_c - self.warm_reference_temp_c) * self.warm_penalty_pct_per_c / 100.0

        return min(factor, self.max_weather_adjustment_factor)

    # ------------------------------------------------------------------
    def estimate_vehicle(
        self,
        vehicle_id: str,
        telemetry_rows: List[sqlite3.Row],
        kwh_per_km: Optional[float] = None,
        ambient_temp_c: Optional[float] = None,
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
        df.to_dict(orient="records") instead of re-querying the DB.

        ambient_temp_c (Feature 8): optional live temperature for this
        vehicle's region (models/weather_client.py). When given,
        kwh_per_km is adjusted by _weather_adjustment_factor before
        computing range. Omit it (the default, None) to get exactly the
        original weather-agnostic estimate — every existing caller that
        doesn't pass this argument is unaffected."""
        base_rate = kwh_per_km if kwh_per_km is not None else self.kwh_per_km
        adjustment = self._weather_adjustment_factor(ambient_temp_c)
        rate = base_rate * adjustment

        if not telemetry_rows:
            return RangeEstimate(
                vehicle_id=vehicle_id, latest_cycle=None, soc_pct=None,
                capacity_kwh_remaining=None, kwh_per_km=rate,
                estimated_range_km=None, at_risk_of_stranding=False,
                ambient_temp_c=ambient_temp_c, weather_adjustment_factor=adjustment,
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
            ambient_temp_c=ambient_temp_c,
            weather_adjustment_factor=adjustment,
        )

    # ------------------------------------------------------------------
    # Live-weather convenience wrappers (Feature 8) — the ONLY methods in
    # this module that touch ingestion/db.py or a WeatherClient.
    # estimate_vehicle() above stays a pure function of its inputs,
    # exactly as before, for every existing caller/test that doesn't pass
    # a weather_client.
    # ------------------------------------------------------------------
    def _latest_location_for_vehicle(
        self, conn: sqlite3.Connection, vehicle_id: str
    ) -> Optional[Tuple[float, float]]:
        from ingestion.db import get_commands_for_vehicle

        rows = get_commands_for_vehicle(conn, vehicle_id)
        if not rows:
            return None
        latest = rows[-1]
        return float(latest["latitude"]), float(latest["longitude"])

    def estimate_vehicle_live(
        self,
        conn: sqlite3.Connection,
        vehicle_id: str,
        weather_client: Optional["object"] = None,
        depot_locations: Tuple[Tuple[float, float], ...] = DEFAULT_DEPOT_LOCATIONS,
        kwh_per_km: Optional[float] = None,
    ) -> RangeEstimate:
        """Same as estimate_vehicle(), but fetches the vehicle's own
        telemetry AND (when weather_client is given) its live regional
        ambient temperature — via its last known command GPS mapped to
        the nearest depot, since telemetry rows carry no GPS of their
        own. If the vehicle has no command history yet, weather_client
        is None, or the live lookup fails for any reason, this degrades
        to exactly estimate_vehicle()'s original weather-agnostic
        behaviour (ambient_temp_c=None -> adjustment factor 1.0).

        weather_client: any object exposing
        get_current_temperature_c(lat, lon) -> Optional[float] — typed as
        a plain object rather than importing models.weather_client.WeatherClient
        directly, so this module has no hard import-time dependency on
        that file (or on `requests`) for callers who never pass one."""
        from ingestion.db import get_telemetry_for_vehicle

        telemetry_rows = get_telemetry_for_vehicle(conn, vehicle_id)

        ambient_temp_c = None
        if weather_client is not None:
            location = self._latest_location_for_vehicle(conn, vehicle_id)
            if location is not None:
                depot_lat, depot_lon = _nearest_depot(location[0], location[1], depot_locations)
                ambient_temp_c = weather_client.get_current_temperature_c(depot_lat, depot_lon)

        return self.estimate_vehicle(
            vehicle_id, telemetry_rows, kwh_per_km=kwh_per_km, ambient_temp_c=ambient_temp_c
        )

    # ------------------------------------------------------------------
    def estimate_fleet(
        self,
        conn: sqlite3.Connection,
        weather_client: Optional["object"] = None,
        depot_locations: Tuple[Tuple[float, float], ...] = DEFAULT_DEPOT_LOCATIONS,
    ) -> pd.DataFrame:
        """weather_client: see estimate_vehicle_live()'s docstring. Omit
        it (the default, None) to get exactly the original weather-
        agnostic fleet estimate — every existing caller/test that calls
        estimate_fleet(conn) with no second argument is unaffected."""
        from ingestion.db import get_all_vehicle_ids

        rows = [
            self.estimate_vehicle_live(
                conn, vid, weather_client=weather_client, depot_locations=depot_locations
            ).to_dict()
            for vid in get_all_vehicle_ids(conn)
        ]
        return pd.DataFrame(rows, columns=[
            "vehicle_id", "latest_cycle", "soc_pct", "capacity_kwh_remaining",
            "kwh_per_km", "estimated_range_km", "at_risk_of_stranding",
            "ambient_temp_c", "weather_adjustment_factor",
        ])


if __name__ == "__main__":
    # Standalone sanity check, same pattern as rul_model.py/charging_analyzer.py.
    # Exercises both the weather-agnostic path and the live-weather path
    # (best-effort — if models/weather_client.py can't reach the network in
    # this environment, get_current_temperature_c() returns None and the
    # live estimate degrades to identical numbers, which is itself the
    # behaviour worth confirming here).
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
    print("Weather-agnostic estimate:")
    print(result_df.to_string(index=False))
    print(f"\n{int(result_df['at_risk_of_stranding'].sum())} vehicle(s) at risk of stranding")

    try:
        from models.weather_client import WeatherClient

        weather_client = WeatherClient()
        live_df = estimator.estimate_fleet(conn, weather_client=weather_client)
        print("\nLive-weather-adjusted estimate:")
        print(live_df[["vehicle_id", "ambient_temp_c", "weather_adjustment_factor",
                        "estimated_range_km"]].to_string(index=False))
    except Exception as e:
        print(f"\n(Skipping live-weather demo: {e})")

    conn.close()
    os.remove(test_db)