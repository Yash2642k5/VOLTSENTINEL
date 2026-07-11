"""
tests/test_anomaly_detector.py

Validates models/anomaly_detector.py against simulator ground truth
(is_attack, thermal_event_flag) — used ONLY here, for testing, exactly
as the project doc requires (the detector itself must never read
these columns in real logic).

This test suite is also what caught the real index-misalignment bug
during development (recall/precision collapsed to ~0.27 before the
fix) — these assertions exist specifically to catch that class of bug
returning.

Run from the project root:
    pytest tests/test_anomaly_detector.py -v
"""

import math

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.maintenance_generator import MaintenanceGenerator
from simulator.attack_injector import AttackInjector
from models.anomaly_detector import AnomalyDetector, _haversine_km, _distance_to_nearest_depot


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=10, num_cycles=100, random_seed=55)


@pytest.fixture(scope="module")
def simulated_data(small_config):
    tgen = TelemetryGenerator(small_config)
    telem_df = tgen.generate_fleet()
    bounds = tgen.get_vehicle_time_bounds(telem_df)

    mgen = MaintenanceGenerator(small_config)
    tickets_df = mgen.generate_fleet_tickets(bounds)

    ainj = AttackInjector(small_config)
    commands_df = ainj.generate_command_stream(bounds, tickets_df)

    return telem_df, tickets_df, commands_df


@pytest.fixture(scope="module")
def detector():
    return AnomalyDetector()


@pytest.fixture(scope="module")
def command_anomaly_result(simulated_data, detector):
    _, _, commands_df = simulated_data
    result = detector.detect_command_anomalies(commands_df.drop(columns=["is_attack"]))
    # Reattach ground truth by index (validates the index-alignment fix holds)
    result = result.copy()
    result["is_attack"] = commands_df["is_attack"]
    return result


@pytest.fixture(scope="module")
def thermal_anomaly_result(simulated_data, detector):
    telem_df, _, _ = simulated_data
    result = detector.detect_thermal_anomalies(telem_df.drop(columns=["thermal_event_flag"]))
    result = result.copy()
    result["thermal_event_flag"] = telem_df["thermal_event_flag"]
    return result


# ----------------------------------------------------------------------
# Haversine distance helper
# ----------------------------------------------------------------------
class TestHaversine:
    def test_zero_distance_same_point(self):
        assert _haversine_km(12.97, 77.59, 12.97, 77.59) == pytest.approx(0, abs=1e-6)

    def test_known_distance_bengaluru_to_delhi(self):
        # Approx real-world great-circle distance is ~1740 km
        dist = _haversine_km(12.9716, 77.5946, 28.7041, 77.1025)
        assert 1600 < dist < 1900

    def test_nearest_depot_picks_closest(self):
        depots = ((12.9716, 77.5946), (28.7041, 77.1025), (19.0760, 72.8777))
        # A point very close to the Bengaluru depot
        dist = _distance_to_nearest_depot(12.9720, 77.5950, depots)
        assert dist < 1.0


# ----------------------------------------------------------------------
# Command anomaly detection — the core correctness suite
# ----------------------------------------------------------------------
class TestCommandAnomalies:
    def test_no_ticket_flag_matches_ticket_id_null(self, command_anomaly_result):
        expected = command_anomaly_result["ticket_id"].isnull()
        assert (command_anomaly_result["no_ticket_flag"] == expected).all()

    def test_legit_commands_never_flagged_no_ticket(self, command_anomaly_result):
        legit = command_anomaly_result[command_anomaly_result["is_attack"] == False]  # noqa: E712
        assert not legit["no_ticket_flag"].any()

    def test_all_attacks_have_at_least_one_signal(self, command_anomaly_result):
        """Every injected attack must be caught by at least one signal —
        primarily no_ticket_flag, since attacks are always ticketless."""
        attacks = command_anomaly_result[command_anomaly_result["is_attack"] == True]  # noqa: E712
        assert (attacks["signal_count"] >= 1).all()

    def test_no_false_positives_on_legitimate_commands(self, command_anomaly_result):
        """Legit commands (real ticket, depot GPS) should never be flagged
        suspicious — this is the assertion that would have caught the
        index-misalignment bug found during development."""
        legit = command_anomaly_result[command_anomaly_result["is_attack"] == False]  # noqa: E712
        assert (legit["signal_count"] == 0).all()

    def test_perfect_recall_and_precision_on_ground_truth(self, command_anomaly_result):
        is_suspicious = command_anomaly_result["signal_count"] >= 1
        is_attack = command_anomaly_result["is_attack"] == True  # noqa: E712
        tp = int((is_suspicious & is_attack).sum())
        fn = int((~is_suspicious & is_attack).sum())
        fp = int((is_suspicious & ~is_attack).sum())

        assert is_attack.sum() > 0, "test fixture produced no attacks — test is vacuous"
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        assert recall == 1.0
        assert precision == 1.0

    def test_severity_labels_match_signal_count(self, command_anomaly_result):
        expected = {0: "none", 1: "low", 2: "medium", 3: "high"}
        for _, row in command_anomaly_result.iterrows():
            assert row["severity"] == expected[row["signal_count"]]

    def test_empty_input_returns_empty_with_columns(self, detector):
        empty = pd.DataFrame(columns=["vehicle_id", "timestamp", "latitude", "longitude", "ticket_id"])
        result = detector.detect_command_anomalies(empty)
        assert len(result) == 0
        assert "no_ticket_flag" in result.columns


# ----------------------------------------------------------------------
# Thermal anomaly detection
# ----------------------------------------------------------------------
class TestThermalAnomalies:
    def test_all_true_thermal_events_are_flagged(self, thermal_anomaly_result):
        """Full recall: every simulator-injected thermal event must be caught.
        (Precision is expected to be looser — the detector flags any reading
        above threshold, which is broader but not wrong; see project notes.)"""
        true_events = thermal_anomaly_result[thermal_anomaly_result["thermal_event_flag"] == True]  # noqa: E712
        assert len(true_events) > 0, "test fixture produced no thermal events — test is vacuous"
        assert true_events["thermal_anomaly"].all()

    def test_critical_flag_implies_sustained_flag(self, thermal_anomaly_result):
        """Critical temp threshold is higher than the safe threshold, so
        anything critical must also be above the safe threshold."""
        critical = thermal_anomaly_result[thermal_anomaly_result["critical_temp_flag"]]
        if len(critical) > 0:
            assert critical["sustained_high_temp_flag"].all()

    def test_sustained_flag_matches_threshold(self, thermal_anomaly_result, detector):
        expected = thermal_anomaly_result["temperature_c"] >= detector.safe_temp_c
        assert (thermal_anomaly_result["sustained_high_temp_flag"] == expected).all()

    def test_empty_input_returns_empty_with_columns(self, detector):
        empty = pd.DataFrame(columns=["vehicle_id", "cycle", "timestamp", "temperature_c"])
        result = detector.detect_thermal_anomalies(empty)
        assert len(result) == 0
        assert "thermal_anomaly" in result.columns


# ----------------------------------------------------------------------
# Isolation Forest (secondary/optional signal)
# ----------------------------------------------------------------------
class TestIsolationForest:
    def test_runs_without_error_on_real_data(self, simulated_data, detector):
        telem_df, _, _ = simulated_data
        result = detector.fit_isolation_forest_scores(telem_df)
        assert "isolation_forest_outlier" in result.columns
        assert result["isolation_forest_outlier"].dtype == bool

    def test_too_few_rows_returns_empty_flag_column(self, detector):
        small = pd.DataFrame({
            "temperature_c": [30, 31, 32],
            "voltage": [51, 51, 51],
            "soc_pct": [80, 79, 78],
            "dod_pct": [50, 51, 52],
        })
        result = detector.fit_isolation_forest_scores(small)
        assert "isolation_forest_outlier" in result.columns