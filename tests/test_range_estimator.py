"""
tests/test_range_estimator.py

Validates models/range_estimator.py: hand-checkable exact-number
computation, the low-SoC / low-range stranding-risk threshold logic,
edge cases, and a DB round-trip against real simulator data — mirroring
the existing style of tests/test_rul_model.py and
tests/test_charging_analyzer.py.

Run from the project root:
    pytest tests/test_range_estimator.py -v
"""

import os

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator

from models.range_estimator import RangeEstimator, RangeEstimate


@pytest.fixture(scope="module")
def estimator():
    return RangeEstimator(kwh_per_km=0.06, low_range_threshold_km=15.0, low_soc_threshold_pct=20.0)


def _row(cycle, soc_pct, capacity_kwh=3.0):
    return {"cycle": cycle, "soc_pct": soc_pct, "capacity_kwh": capacity_kwh}


# ----------------------------------------------------------------------
# Hand-checkable exact computation
# ----------------------------------------------------------------------
class TestExactComputation:
    def test_capacity_remaining_exact(self, estimator):
        # 3.0 kWh full-charge capacity at 50% SoC -> 1.5 kWh remaining
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.capacity_kwh_remaining == pytest.approx(1.5)

    def test_estimated_range_exact(self, estimator):
        # 1.5 kWh remaining / 0.06 kWh/km = 25.0 km
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.estimated_range_km == pytest.approx(25.0)

    def test_only_latest_row_is_used(self, estimator):
        rows = [_row(1, 90.0, capacity_kwh=3.0), _row(2, 40.0, capacity_kwh=2.9)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.soc_pct == pytest.approx(40.0)
        assert result.latest_cycle == 2

    def test_custom_kwh_per_km_override(self, estimator):
        rows = [_row(1, 100.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, kwh_per_km=0.1)
        assert result.kwh_per_km == pytest.approx(0.1)
        assert result.estimated_range_km == pytest.approx(30.0)


# ----------------------------------------------------------------------
# Stranding-risk threshold logic
# ----------------------------------------------------------------------
class TestStrandingRisk:
    def test_low_soc_flags_at_risk_even_with_ample_range(self, estimator):
        # SoC below threshold (20%) should flag regardless of raw range math
        rows = [_row(1, 15.0, capacity_kwh=10.0)]  # range would be huge otherwise
        result = estimator.estimate_vehicle("V1", rows)
        assert result.at_risk_of_stranding is True

    def test_low_range_flags_at_risk_even_with_ok_soc(self, estimator):
        # SoC above threshold, but tiny capacity -> range below 15km threshold
        rows = [_row(1, 50.0, capacity_kwh=0.5)]  # 0.25 kWh / 0.06 = ~4.2 km
        result = estimator.estimate_vehicle("V1", rows)
        assert result.at_risk_of_stranding is True

    def test_healthy_vehicle_not_at_risk(self, estimator):
        rows = [_row(1, 80.0, capacity_kwh=3.0)]  # 2.4kWh / 0.06 = 40km, SoC 80%
        result = estimator.estimate_vehicle("V1", rows)
        assert result.at_risk_of_stranding is False

    def test_boundary_soc_exactly_at_threshold_flags(self, estimator):
        rows = [_row(1, 20.0, capacity_kwh=10.0)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.at_risk_of_stranding is True

    def test_boundary_range_exactly_at_threshold_flags(self, estimator):
        # 0.9 kWh / 0.06 = 15.0 km exactly
        rows = [_row(1, 50.0, capacity_kwh=1.8)]
        result = estimator.estimate_vehicle("V1", rows)
        assert result.estimated_range_km == pytest.approx(15.0)
        assert result.at_risk_of_stranding is True


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_rows_returns_none_fields_and_not_at_risk(self, estimator):
        result = estimator.estimate_vehicle("V-EMPTY", [])
        assert result.soc_pct is None
        assert result.estimated_range_km is None
        assert result.at_risk_of_stranding is False

    def test_zero_kwh_per_km_returns_none_range_not_a_crash(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows, kwh_per_km=0.0)
        assert result.estimated_range_km is None

    def test_result_is_serializable_dict(self, estimator):
        rows = [_row(1, 50.0, capacity_kwh=3.0)]
        result = estimator.estimate_vehicle("V1", rows)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["vehicle_id"] == "V1"


# ----------------------------------------------------------------------
# Fleet-level integration (DB round trip)
# ----------------------------------------------------------------------
def test_estimate_fleet_via_db():
    from ingestion.db import get_connection, init_db, insert_telemetry_batch
    from ingestion.schemas import TelemetryReading

    cfg = SimulatorConfig(fleet_size=5, num_cycles=40, random_seed=13)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()

    test_db = os.path.join("data", "test_range_estimator.db")
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
    conn.close()
    os.remove(test_db)

    assert len(result_df) == cfg.fleet_size
    assert set(result_df["vehicle_id"]) == set(telem_df["vehicle_id"].unique())
    assert result_df["at_risk_of_stranding"].dtype == bool
    # Every vehicle has real telemetry, so estimates should never be null here
    assert result_df["estimated_range_km"].notnull().all()
    assert result_df["soc_pct"].between(0, 100).all()