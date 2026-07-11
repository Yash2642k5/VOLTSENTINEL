"""
tests/test_charging_analyzer.py

Validates models/charging_analyzer.py: per-vehicle stat computation
against hand-checkable synthetic data, stress-trend detection logic,
fleet-baseline computation, suggested_policy triggers, and edge cases.

Not part of the originally documented folder_structure.py test list —
added on request to cover Phase 3's remaining two files.

Run from the project root:
    pytest tests/test_charging_analyzer.py -v
"""

import math
import os

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from models.charging_analyzer import ChargingAnalyzer, ChargingProfile, DEFAULT_HIGH_DOD_THRESHOLD_PCT


@pytest.fixture(scope="module")
def analyzer():
    return ChargingAnalyzer()


def _row(cycle, is_fast_charge, dod_pct):
    return {"cycle": cycle, "is_fast_charge": is_fast_charge, "dod_pct": dod_pct}


# ----------------------------------------------------------------------
# Hand-checkable synthetic data — exact numbers, not "looks about right"
# ----------------------------------------------------------------------
class TestExactComputation:
    def test_fast_charge_frequency_exact(self, analyzer):
        # 10 rows, 3 fast-charge -> exactly 30%
        rows = [_row(i, i < 3, 50.0) for i in range(10)]
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.fast_charge_frequency_pct == pytest.approx(30.0)

    def test_mean_dod_exact(self, analyzer):
        rows = [_row(i, False, 60.0) for i in range(10)]
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.mean_dod_pct == pytest.approx(60.0)

    def test_high_dod_frequency_exact(self, analyzer):
        # threshold default 85.0; 4 out of 10 rows at 90% -> 40%
        rows = [_row(i, False, 90.0 if i < 4 else 50.0) for i in range(10)]
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.high_dod_frequency_pct == pytest.approx(40.0)

    def test_fast_charge_vs_baseline_delta(self, analyzer):
        rows = [_row(i, i < 5, 50.0) for i in range(10)]  # 50% fast charge
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=20.0)
        assert profile.fast_charge_vs_baseline_pct == pytest.approx(30.0)

    def test_charge_stress_score_formula(self, analyzer):
        """Score = 0.4*fast_freq + 0.4*min(dod,100) + 0.2*high_dod_freq — verify
        the documented weighting is actually what's implemented."""
        rows = [_row(i, i < 5, 90.0) for i in range(10)]  # 50% fast, 90% dod, 100% high-dod
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        expected = 0.4 * 50.0 + 0.4 * 90.0 + 0.2 * 100.0
        assert profile.charge_stress_score == pytest.approx(expected, abs=0.01)


# ----------------------------------------------------------------------
# Stress trend detection
# ----------------------------------------------------------------------
class TestStressTrend:
    def test_increasing_trend_detected(self, analyzer):
        # early window: no fast charge, low dod. recent window: all fast charge, high dod.
        w = analyzer.trend_window
        rows = (
            [_row(i, False, 40.0) for i in range(w)]
            + [_row(i, False, 40.0) for i in range(w, 50)]  # middle padding
            + [_row(i, True, 95.0) for i in range(50, 50 + w)]
        )
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.stress_trend == "increasing"

    def test_decreasing_trend_detected(self, analyzer):
        w = analyzer.trend_window
        rows = (
            [_row(i, True, 95.0) for i in range(w)]
            + [_row(i, False, 40.0) for i in range(w, 50)]
            + [_row(i, False, 30.0) for i in range(50, 50 + w)]
        )
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.stress_trend == "decreasing"

    def test_stable_trend_detected(self, analyzer):
        w = analyzer.trend_window
        rows = [_row(i, i % 3 == 0, 55.0) for i in range(w * 3)]
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.stress_trend == "stable"

    def test_insufficient_data_below_two_windows(self, analyzer):
        w = analyzer.trend_window
        rows = [_row(i, False, 50.0) for i in range(w)]  # exactly one window, not two
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.stress_trend == "insufficient_data"


# ----------------------------------------------------------------------
# suggested_policy triggers
# ----------------------------------------------------------------------
class TestSuggestedPolicy:
    def test_high_fast_charge_triggers_cap_suggestion(self, analyzer):
        rows = [_row(i, True, 50.0) for i in range(20)]  # 100% fast charge
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.suggested_policy is not None
        assert "fast-charge" in profile.suggested_policy

    def test_high_dod_triggers_limit_suggestion(self, analyzer):
        rows = [_row(i, False, 95.0) for i in range(20)]  # above default threshold (85)
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.suggested_policy is not None
        assert "depth-of-discharge" in profile.suggested_policy

    def test_no_concerns_gives_none(self, analyzer):
        rows = [_row(i, False, 50.0) for i in range(20)]  # mild, stable
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.suggested_policy is None

    def test_multiple_concerns_joined(self, analyzer):
        w = analyzer.trend_window
        rows = (
            [_row(i, False, 40.0) for i in range(w)]
            + [_row(i, True, 95.0) for i in range(w, w + 30)]  # high fast-charge + high dod + increasing
        )
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        assert profile.suggested_policy is not None
        assert ";" in profile.suggested_policy  # more than one suggestion joined


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_rows_returns_zeroed_profile(self, analyzer):
        profile = analyzer.analyze_vehicle("V-EMPTY", [], fleet_baseline_pct=25.0)
        assert profile.total_cycles == 0
        assert profile.fast_charge_frequency_pct == 0.0
        assert profile.stress_trend == "insufficient_data"

    def test_profile_is_serializable_dict(self, analyzer):
        rows = [_row(i, False, 50.0) for i in range(10)]
        profile = analyzer.analyze_vehicle("V1", rows, fleet_baseline_pct=25.0)
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["vehicle_id"] == "V1"


# ----------------------------------------------------------------------
# Fleet-level integration (DB round trip)
# ----------------------------------------------------------------------
def test_analyze_fleet_via_db():
    from ingestion.db import get_connection, init_db, insert_telemetry_batch
    from ingestion.schemas import TelemetryReading

    cfg = SimulatorConfig(fleet_size=5, num_cycles=60, random_seed=8)
    tgen = TelemetryGenerator(cfg)
    telem_df = tgen.generate_fleet()

    test_db = os.path.join("data", "test_charging_analyzer.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    conn = get_connection(test_db)
    init_db(conn)

    readings = [TelemetryReading(**r) for r in telem_df.to_dict(orient="records")]
    insert_telemetry_batch(conn, readings)

    analyzer = ChargingAnalyzer()
    profiles_df = analyzer.analyze_fleet(conn)
    conn.close()
    os.remove(test_db)

    assert len(profiles_df) == cfg.fleet_size
    assert set(profiles_df["vehicle_id"]) == set(telem_df["vehicle_id"].unique())
    # Fleet baseline should be identical across all rows (computed once for the fleet)
    assert profiles_df["fleet_fast_charge_baseline_pct"].nunique() == 1
    # Baseline should roughly match config's fast_charge_probability (30%), generously bounded
    baseline = profiles_df["fleet_fast_charge_baseline_pct"].iloc[0]
    assert 15.0 < baseline < 45.0