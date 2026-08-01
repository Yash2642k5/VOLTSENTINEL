"""
Using Open-Meteo (https://open-meteo.com) for current temperature by coordinate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DEFAULT_CACHE_TTL_SECONDS = 600
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
            return None

    def clear_cache(self) -> None:
        self._cache.clear()


if __name__ == "__main__":
    client = WeatherClient()
    temp = client.get_current_temperature_c(12.9716, 77.5946)
    print(f"Bengaluru current temperature: {temp}°C" if temp is not None else "Weather unavailable")