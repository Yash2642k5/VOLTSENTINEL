"""Asset Registry tab — make/model/VIN/purchase date/warranty status per vehicle."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from dashboard.utils import risk_level_label


def _warranty_status(warranty_expiry: str, now: datetime) -> str:
    try:
        expiry = pd.to_datetime(warranty_expiry)
    except (ValueError, TypeError):
        return "Unknown"
    if expiry.tzinfo is not None:
        expiry = expiry.tz_localize(None)
    return "Active" if expiry >= now else "Expired"


def build_asset_registry_view(
    metadata_df: pd.DataFrame, profile_df: pd.DataFrame, now: datetime | None = None,
) -> pd.DataFrame:
    if metadata_df.empty:
        return metadata_df

    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    df = metadata_df.copy()

    if not profile_df.empty and "overall_risk_level" in profile_df.columns:
        df = df.merge(
            profile_df[["vehicle_id", "overall_risk_level"]], on="vehicle_id", how="left"
        )
    else:
        df["overall_risk_level"] = None

    df["purchase_date_parsed"] = pd.to_datetime(df["purchase_date"]).dt.tz_localize(None)
    df["warranty_expiry_parsed"] = pd.to_datetime(df["warranty_expiry_date"]).dt.tz_localize(None)
    df["age_days"] = (now - df["purchase_date_parsed"]).dt.days
    df["warranty_status"] = df["warranty_expiry_date"].map(lambda v: _warranty_status(v, now))
    df = df.sort_values(["warranty_status", "age_days"], ascending=[True, False])

    return pd.DataFrame({
        "Vehicle ID": df["vehicle_id"],
        "Make": df["make"],
        "Model": df["model"],
        "VIN": df["vin"],
        "Purchase Date": df["purchase_date_parsed"].dt.strftime("%Y-%m-%d"),
        "Age (days)": df["age_days"],
        "Warranty Expiry": df["warranty_expiry_parsed"].dt.strftime("%Y-%m-%d"),
        "Warranty Status": df["warranty_status"],
        "Risk Level": df["overall_risk_level"].map(
            lambda v: risk_level_label(v if pd.notnull(v) else None)
        ),
    })


def render_asset_registry(conn, profile_df: pd.DataFrame) -> None:
    from dashboard.utils import get_all_vehicle_metadata

    st.subheader("Asset Registry")
    st.caption("Make, model, VIN, purchase date, and warranty status for every vehicle in the fleet.")

    metadata_df = get_all_vehicle_metadata(conn)
    if metadata_df.empty:
        st.info("No asset-registry entries yet — reseed the DB to populate the `vehicles` table.")
        return

    view = build_asset_registry_view(metadata_df, profile_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vehicles", len(view))
    col2.metric("Active Warranty", int((view["Warranty Status"] == "Active").sum()))
    col3.metric("Expired Warranty", int((view["Warranty Status"] == "Expired").sum()))

    st.dataframe(view, use_container_width=True, hide_index=True)
