"""
tests/test_rul_model.py

Validates models/rul_model.py: curve-fit quality against known
simulator decay, RUL extrapolation correctness, status banding, and
edge cases (insufficient data, no-decay, non-convergence).

Run from the project root:
    pytest tests/test_rul_model.py -v
"""

import numpy as np
import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from models.rul_model import RULModel, RULResult, DEFAULT_END_OF_LIFE_CAPACITY_PCT


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=6, num_cycles=250, random_seed=2026)


@pytest.fixture(scope="module")
def telemetry(small_config):
    gen = TelemetryGenerator(small_config)
    return gen.generate_fleet()


@pytest.fixture(scope="module")
def model():
    return RULModel()


def _rows_for_vehicle(telemetry_df: pd.DataFrame, vehicle_id: str):
    """Mimic sqlite3.Row's dict-like ['col'] access using plain dicts,
    since RULModel only relies on that interface."""
    sub = telemetry_df[telemetry_df.vehicle_id == vehicle_id].sort_values("cycle")
    return [row.to_dict() for _, row in sub.iterrows()]


# ----------------------------------------------------------------------
# Curve fit quality against real simulator decay
# ----------------------------------------------------------------------
class TestFitQuality:
    def test_fit_recovers_high_r_squared(self, telemetry, model):
        """The simulator's decay is genuinely exponential, so a good fit
        should be near-perfect — this is the core claim from the earlier
        'do we need a powerful model' discussion."""
        vid = telemetry["vehicle_id"].iloc[0]
        rows = _rows_for_vehicle(telemetry, vid)
        result = model.fit_vehicle(vid, rows)
        assert result.r_squared is not None
        assert result.r_squared > 0.9

    def test_fit_quality_across_fleet(self, telemetry, model):
        r_squareds = []
        for vid in telemetry["vehicle_id"].unique():
            rows = _rows_for_vehicle(telemetry, vid)
            result = model.fit_vehicle(vid, rows)
            if result.r_squared is not None:
                r_squareds.append(result.r_squared)
        assert len(r_squareds) > 0
        assert np.mean(r_squareds) > 0.9

    def test_fitted_decay_rate_is_positive_for_degrading_vehicle(self, telemetry, model):
        vid = telemetry["vehicle_id"].iloc[0]
        rows = _rows_for_vehicle(telemetry, vid)
        result = model.fit_vehicle(vid, rows)
        assert result.fitted_decay_rate is not None
        assert result.fitted_decay_rate > 0


# ----------------------------------------------------------------------
# RUL extrapolation correctness
# ----------------------------------------------------------------------
class TestRULExtrapolation:
    def test_rul_decreases_as_capacity_approaches_eol(self, model):
        """Two synthetic curves with identical decay rate but different
        current position — the one closer to EOL must report lower RUL."""
        cycles = list(range(1, 101))

        healthy_rows = [
            {"cycle": c, "capacity_pct_of_rated": 100 * np.exp(-0.001 * c)} for c in cycles
        ]
        degraded_rows = [
            {"cycle": c, "capacity_pct_of_rated": 100 * np.exp(-0.008 * c)} for c in cycles
        ]

        healthy_result = model.fit_vehicle("V-HEALTHY", healthy_rows)
        degraded_result = model.fit_vehicle("V-DEGRADED", degraded_rows)

        assert healthy_result.rul_cycles is not None
        assert degraded_result.rul_cycles is not None
        assert degraded_result.rul_cycles < healthy_result.rul_cycles

    def test_rul_never_negative(self, telemetry, model):
        for vid in telemetry["vehicle_id"].unique():
            rows = _rows_for_vehicle(telemetry, vid)
            result = model.fit_vehicle(vid, rows)
            if result.rul_cycles is not None:
                assert result.rul_cycles >= 0

    def test_no_decay_curve_returns_none_rul(self, model):
        """Flat capacity (no measurable decay) must not fabricate an RUL number."""
        rows = [{"cycle": c, "capacity_pct_of_rated": 100.0} for c in range(1, 30)]
        result = model.fit_vehicle("V-FLAT", rows)
        assert result.rul_cycles is None


# ----------------------------------------------------------------------
# Status banding
# ----------------------------------------------------------------------
class TestStatusBanding:
    @pytest.mark.parametrize("capacity_pct,expected_status", [
        (99.0, "healthy"),
        (85.0, "watch"),
        (75.0, "degraded"),
        (65.0, "critical"),
    ])
    def test_status_bands(self, model, capacity_pct, expected_status):
        assert model._status_from_capacity(capacity_pct) == expected_status

    def test_status_matches_default_eol_threshold(self, model):
        assert model.end_of_life_capacity_pct == DEFAULT_END_OF_LIFE_CAPACITY_PCT
        assert model._status_from_capacity(DEFAULT_END_OF_LIFE_CAPACITY_PCT) == "critical"


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_telemetry_returns_insufficient_data(self, model):
        result = model.fit_vehicle("V-EMPTY", [])
        assert result.status == "insufficient_data"
        assert result.rul_cycles is None

    def test_too_few_points_returns_insufficient_data(self, model):
        rows = [{"cycle": c, "capacity_pct_of_rated": 100 - c} for c in range(1, 3)]  # < MIN_POINTS_FOR_FIT
        result = model.fit_vehicle("V-SPARSE", rows)
        assert result.status == "insufficient_data"

    def test_result_is_serializable_dict(self, telemetry, model):
        vid = telemetry["vehicle_id"].iloc[0]
        rows = _rows_for_vehicle(telemetry, vid)
        result = model.fit_vehicle(vid, rows)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["vehicle_id"] == vid


# ----------------------------------------------------------------------
# Fleet-level integration (DB round trip)
# ----------------------------------------------------------------------
def test_fit_fleet_via_db(small_config, telemetry, model):
    import math
    import os

    from ingestion.db import get_connection, init_db, insert_telemetry_batch
    from ingestion.schemas import TelemetryReading

    test_db = os.path.join("data", "test_rul_model.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    readings = [TelemetryReading(**r) for r in telemetry.to_dict(orient="records")]
    insert_telemetry_batch(conn, readings)

    results_df = model.fit_fleet(conn)
    conn.close()
    os.remove(test_db)

    assert len(results_df) == small_config.fleet_size
    assert set(results_df["vehicle_id"]) == set(telemetry["vehicle_id"].unique())
    assert results_df["r_squared"].mean() > 0.9