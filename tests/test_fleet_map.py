"""
tests/test_fleet_map.py

Validates dashboard/components/fleet_map.py's pure data-prep functions,
focused on live-vehicle-state integration (position/activity from
scripts/live_feed.py taking precedence over command-derived locations).

Run from the project root:
    pytest tests/test_fleet_map.py -v
"""

import pandas as pd

from dashboard.components.fleet_map import (
    ACTIVITY_STATUS_LABELS,
    build_fleet_table_view,
    build_vehicle_scatter_data,
    _merge_live_locations,
)


def _profile_row(vehicle_id, overall_risk_level="minimal", **overrides):
    row = {
        "vehicle_id": vehicle_id, "status": "healthy", "overall_risk_level": overall_risk_level,
        "rul_cycles": 100, "thermal_anomaly_count": 0, "max_security_severity": "none",
        "charge_stress_score": 10,
    }
    row.update(overrides)
    return row


class TestMergeLiveLocations:
    def test_no_live_state_falls_back_to_commands(self):
        locations_df = pd.DataFrame([{"vehicle_id": "EVR-0001", "latitude": 1.0, "longitude": 2.0}])
        result = _merge_live_locations(locations_df, None)
        assert result.iloc[0]["activity_status"] is None
        assert result.iloc[0]["latitude"] == 1.0

    def test_live_state_takes_precedence_for_shared_vehicle(self):
        locations_df = pd.DataFrame([{"vehicle_id": "EVR-0001", "latitude": 1.0, "longitude": 2.0}])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0001", "latitude": 9.0, "longitude": 8.0, "activity_status": "active"},
        ])
        result = _merge_live_locations(locations_df, live_state_df)
        row = result[result["vehicle_id"] == "EVR-0001"].iloc[0]
        assert (row["latitude"], row["longitude"]) == (9.0, 8.0)
        assert row["activity_status"] == "active"

    def test_vehicle_only_in_live_state_still_appears(self):
        """A vehicle with zero BMS command history should still show up
        on the map once the live feed has reported a position for it."""
        locations_df = pd.DataFrame(columns=["vehicle_id", "latitude", "longitude"])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0002", "latitude": 5.0, "longitude": 6.0, "activity_status": "inactive"},
        ])
        result = _merge_live_locations(locations_df, live_state_df)
        assert list(result["vehicle_id"]) == ["EVR-0002"]

    def test_vehicle_only_in_commands_still_appears(self):
        locations_df = pd.DataFrame([{"vehicle_id": "EVR-0003", "latitude": 1.0, "longitude": 2.0}])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0099", "latitude": 5.0, "longitude": 6.0, "activity_status": "active"},
        ])
        result = _merge_live_locations(locations_df, live_state_df)
        assert set(result["vehicle_id"]) == {"EVR-0003", "EVR-0099"}
        row = result[result["vehicle_id"] == "EVR-0003"].iloc[0]
        assert row["activity_status"] is None


class TestBuildVehicleScatterData:
    def test_inactive_vehicle_gets_dimmer_fill_than_active(self):
        locations_df = pd.DataFrame(columns=["vehicle_id", "latitude", "longitude"])
        profile_df = pd.DataFrame([_profile_row("EVR-0001"), _profile_row("EVR-0002")])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0001", "latitude": 1.0, "longitude": 1.0, "activity_status": "active"},
            {"vehicle_id": "EVR-0002", "latitude": 2.0, "longitude": 2.0, "activity_status": "inactive"},
        ])
        scatter_df = build_vehicle_scatter_data(locations_df, profile_df, live_state_df=live_state_df)
        active_alpha = scatter_df[scatter_df["vehicle_id"] == "EVR-0001"].iloc[0]["fill_color"][3]
        inactive_alpha = scatter_df[scatter_df["vehicle_id"] == "EVR-0002"].iloc[0]["fill_color"][3]
        assert inactive_alpha < active_alpha

    def test_label_includes_activity_status(self):
        locations_df = pd.DataFrame(columns=["vehicle_id", "latitude", "longitude"])
        profile_df = pd.DataFrame([_profile_row("EVR-0001")])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0001", "latitude": 1.0, "longitude": 1.0, "activity_status": "inactive"},
        ])
        scatter_df = build_vehicle_scatter_data(locations_df, profile_df, live_state_df=live_state_df)
        assert ACTIVITY_STATUS_LABELS["inactive"] in scatter_df.iloc[0]["label"]

    def test_no_live_state_still_works(self):
        locations_df = pd.DataFrame([{"vehicle_id": "EVR-0001", "latitude": 1.0, "longitude": 1.0}])
        profile_df = pd.DataFrame([_profile_row("EVR-0001")])
        scatter_df = build_vehicle_scatter_data(locations_df, profile_df)
        assert len(scatter_df) == 1

    def test_empty_everything_returns_empty(self):
        result = build_vehicle_scatter_data(
            pd.DataFrame(columns=["vehicle_id", "latitude", "longitude"]), pd.DataFrame(),
        )
        assert result.empty


class TestBuildFleetTableView:
    def test_activity_column_reflects_live_state(self):
        profile_df = pd.DataFrame([_profile_row("EVR-0001"), _profile_row("EVR-0002")])
        live_state_df = pd.DataFrame([
            {"vehicle_id": "EVR-0001", "activity_status": "active"},
            {"vehicle_id": "EVR-0002", "activity_status": "inactive"},
        ])
        view = build_fleet_table_view(profile_df, live_state_df=live_state_df)
        activity_by_vehicle = dict(zip(view["Vehicle"], view["Activity"]))
        assert activity_by_vehicle["EVR-0001"] == ACTIVITY_STATUS_LABELS["active"]
        assert activity_by_vehicle["EVR-0002"] == ACTIVITY_STATUS_LABELS["inactive"]

    def test_missing_live_state_shows_placeholder(self):
        profile_df = pd.DataFrame([_profile_row("EVR-0001")])
        view = build_fleet_table_view(profile_df)
        assert view.iloc[0]["Activity"] == "—"
