"""
tests/test_asset_registry.py

Validates the Asset Registry feature end to end: AssetGenerator's output
shape, ingestion/db.py's vehicles table round-trip, and
dashboard/components/asset_registry.py's pure view-builder — matching
this project's existing test style of exercising the real pipeline
against a throwaway SQLite file (tests/test_driver_assignments.py,
tests/test_bi_tools.py).

Run from the project root:
    pytest tests/test_asset_registry.py -v
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from simulator.config import SimulatorConfig
from simulator.telemetry_generator import TelemetryGenerator
from simulator.asset_generator import AssetGenerator

from ingestion.db import (
    get_connection, init_db, insert_vehicle_metadata_batch,
    get_vehicle_metadata, get_all_vehicle_metadata, row_counts,
)
from ingestion.schemas import VehicleMetadata

from dashboard.components.asset_registry import build_asset_registry_view

TEST_DB_PATH = os.path.join("data", "test_asset_registry.db")


@pytest.fixture(scope="module")
def small_config():
    return SimulatorConfig(fleet_size=8, num_cycles=20, random_seed=7)


@pytest.fixture(scope="module")
def vehicle_ids(small_config):
    return TelemetryGenerator(small_config).vehicle_ids


@pytest.fixture(scope="module")
def assets_df(small_config, vehicle_ids):
    return AssetGenerator(small_config).generate_fleet_assets(vehicle_ids)


@pytest.fixture(scope="module")
def conn(assets_df):
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    connection = get_connection(TEST_DB_PATH)
    init_db(connection)

    assets = [VehicleMetadata(**r) for r in assets_df.to_dict(orient="records")]
    insert_vehicle_metadata_batch(connection, assets)

    yield connection
    connection.close()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


# ----------------------------------------------------------------------
# AssetGenerator
# ----------------------------------------------------------------------
class TestAssetGenerator:
    def test_one_row_per_vehicle(self, small_config, vehicle_ids, assets_df):
        assert len(assets_df) == small_config.fleet_size
        assert set(assets_df["vehicle_id"]) == set(vehicle_ids)

    def test_vins_are_unique(self, assets_df):
        assert assets_df["vin"].is_unique

    def test_warranty_expiry_after_purchase_date(self, assets_df):
        purchase = pd.to_datetime(assets_df["purchase_date"])
        expiry = pd.to_datetime(assets_df["warranty_expiry_date"])
        assert (expiry > purchase).all()

    def test_reproducible_given_same_seed(self, small_config, vehicle_ids):
        first = AssetGenerator(small_config).generate_fleet_assets(vehicle_ids)
        second = AssetGenerator(small_config).generate_fleet_assets(vehicle_ids)
        pd.testing.assert_frame_equal(first, second)

    def test_validates_against_schema(self, assets_df):
        for row in assets_df.to_dict(orient="records"):
            VehicleMetadata(**row)  # raises on schema mismatch


# ----------------------------------------------------------------------
# ingestion/db.py round-trip
# ----------------------------------------------------------------------
class TestVehiclesTable:
    def test_row_count_matches_fleet_size(self, small_config, conn):
        assert row_counts(conn)["vehicles"] == small_config.fleet_size

    def test_get_vehicle_metadata_known_id(self, vehicle_ids, conn):
        row = get_vehicle_metadata(conn, vehicle_ids[0])
        assert row is not None
        assert row["vehicle_id"] == vehicle_ids[0]
        assert row["make"]
        assert row["model"]

    def test_get_vehicle_metadata_unknown_id_returns_none(self, conn):
        assert get_vehicle_metadata(conn, "EVR-9999") is None

    def test_get_all_vehicle_metadata_returns_full_fleet(self, small_config, conn):
        rows = get_all_vehicle_metadata(conn)
        assert len(rows) == small_config.fleet_size

    def test_reinserting_same_batch_is_idempotent(self, conn, assets_df):
        before = row_counts(conn)["vehicles"]
        assets = [VehicleMetadata(**r) for r in assets_df.to_dict(orient="records")]
        insert_vehicle_metadata_batch(conn, assets)
        assert row_counts(conn)["vehicles"] == before


# ----------------------------------------------------------------------
# dashboard/components/asset_registry.py -- pure view builder
# ----------------------------------------------------------------------
class TestBuildAssetRegistryView:
    def test_empty_metadata_returns_empty(self):
        result = build_asset_registry_view(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_view_has_one_row_per_vehicle(self, small_config, assets_df):
        view = build_asset_registry_view(assets_df, pd.DataFrame())
        assert len(view) == small_config.fleet_size

    def test_warranty_status_reflects_expiry_vs_now(self, vehicle_ids):
        now = datetime(2030, 1, 1)
        metadata_df = pd.DataFrame([
            {
                "vehicle_id": vehicle_ids[0], "make": "Mahindra", "model": "Treo",
                "vin": "1" * 17,
                "purchase_date": "2020-01-01T00:00:00",
                "warranty_expiry_date": "2023-01-01T00:00:00",  # expired well before `now`
            },
            {
                "vehicle_id": vehicle_ids[1], "make": "TVS", "model": "King EV Max",
                "vin": "2" * 17,
                "purchase_date": "2029-01-01T00:00:00",
                "warranty_expiry_date": "2032-01-01T00:00:00",  # active as of `now`
            },
        ])
        view = build_asset_registry_view(metadata_df, pd.DataFrame(), now=now)
        status_by_vehicle = dict(zip(view["Vehicle ID"], view["Warranty Status"]))
        assert status_by_vehicle[vehicle_ids[0]] == "Expired"
        assert status_by_vehicle[vehicle_ids[1]] == "Active"

    def test_merges_risk_level_when_profile_available(self, assets_df):
        vid = assets_df["vehicle_id"].iloc[0]
        profile_df = pd.DataFrame([{"vehicle_id": vid, "overall_risk_level": "high"}])
        view = build_asset_registry_view(assets_df, profile_df)
        row = view[view["Vehicle ID"] == vid].iloc[0]
        assert row["Risk Level"] == "High"

    def test_missing_profile_leaves_risk_level_unknown(self, assets_df):
        view = build_asset_registry_view(assets_df, pd.DataFrame())
        assert (view["Risk Level"] == "Unknown").all()

    def test_merges_reliability_metrics_when_available(self, assets_df):
        vid = assets_df["vehicle_id"].iloc[0]
        reliability_df = pd.DataFrame([
            {"vehicle_id": vid, "ticket_count": 3, "mtbf_hours": 120.5, "mttr_hours": 4.0},
        ])
        view = build_asset_registry_view(assets_df, pd.DataFrame(), reliability_df)
        row = view[view["Vehicle ID"] == vid].iloc[0]
        assert row["Maintenance Events"] == 3
        assert row["MTBF"] == "120.5 hrs"
        assert row["MTTR"] == "4.0 hrs"

    def test_missing_reliability_data_shows_placeholder(self, assets_df):
        view = build_asset_registry_view(assets_df, pd.DataFrame())
        assert (view["MTBF"] == "—").all()
        assert (view["MTTR"] == "—").all()
