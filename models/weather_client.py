"""
models/weather_client.py

Future Roadmap Feature 8 — Weather-Aware Range Estimate.

A thin, network-isolated client for live ambient temperature, consumed
ONLY by models/range_estimator.py as a range-adjustment input — no
other module needs to know weather exists, matching the roadmap's
stated shape for this feature.

Uses Open-Meteo (https://open-meteo.com) for current temperature by
coordinate. Chosen deliberately over a key-gated provider (e.g.
OpenWeatherMap) because it needs NO API key and NO account signup —
matching this project's existing "works with zero configuration, opts
into extra behaviour when configured" philosophy (see
agent/actions.py's SMTP fallback, agent/decision_engine.py's
GEMINI_API_KEY fallback). Open-Meteo's current-weather endpoint is
sourced from national weather services and updated hourly, so this is
real live data, not a mock — swap in OpenWeatherMap or another
provider here later without touching range_estimator.py at all, since
callers only ever see get_current_temperature_c().

Design:
  - get_current_temperature_c() NEVER raises. Any failure (network
    down, timeout, malformed response, rate limit, `requests` not
    installed) returns None — the same "degrade gracefully, never
    break the caller" contract every other optional integration in
    this codebase follows (SMTP, Gemini).
  - A small in-memory, per-coordinate TTL cache is built into the
    client itself (default 10 minutes), not left to callers, since
    ambient temperature changes slowly and the dashboard reruns every
    few seconds (dashboard/utils.py's DEFAULT_CACHE_TTL_SECONDS=8) —
    without this, every rerun would fire one HTTP request per depot
    region.
  - Coordinates are rounded to 2 decimal places (~1.1km) before both
    the cache key and the outbound request, since exact-GPS precision
    doesn't change ambient weather and rounding lets nearby lookups
    (e.g. several vehicles near the same depot) share one cache entry
    and one HTTP call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DEFAULT_CACHE_TTL_SECONDS = 600     # 10 minutes — ambient temp changes slowly
DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _round_coord(value: float) -> float:
    return round(value, 2)


@dataclass
class WeatherClient:
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    _cache: Dict[Tuple[float, float], Tuple[float, Optional[float]]] = field(
        default_factory=dict, repr=False
    )

    def get_current_temperature_c(self, latitude: float, longitude: float) -> Optional[float]:
        """Current ambient temperature in Celsius at the given coordinate,
        or None if it couldn't be determined (network error, timeout,
        malformed response, or provider unavailable). Callers must treat
        None as "no live weather available right now" and fall back to a
        weather-agnostic estimate — never crash on this."""
        key = (_round_coord(latitude), _round_coord(longitude))
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        temp_c = self._fetch(key[0], key[1])
        self._cache[key] = (now, temp_c)
        return temp_c

    def _fetch(self, latitude: float, longitude: float) -> Optional[float]:
        try:
            import requests
        except ImportError:
            return None

        try:
            response = requests.get(
                OPEN_METEO_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m",
                    "timezone": "auto",
                },
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            temp = data.get("current", {}).get("temperature_2m")
            return float(temp) if temp is not None else None
        except Exception:
            # Network error, timeout, bad JSON, unexpected response shape,
            # rate limiting, etc. This integration is additive — a failure
            # here must never break range estimation, only fall back to
            # it being weather-agnostic for this one call.
            return None

    def clear_cache(self) -> None:
        self._cache.clear()


if __name__ == "__main__":
    # Standalone sanity check — hits the real API once for Bengaluru.
    client = WeatherClient()
    temp = client.get_current_temperature_c(12.9716, 77.5946)
    print(f"Bengaluru current temperature: {temp}°C" if temp is not None else "Weather unavailable")