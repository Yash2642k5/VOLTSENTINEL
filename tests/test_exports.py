"""
tests/test_exports.py

Validates dashboard/exports.py: Excel output round-trips back to the
same data via pandas, PDF output is a well-formed PDF byte stream.

Run from the project root:
    pytest tests/test_exports.py -v
"""

from io import BytesIO

import pandas as pd
import pytest

from dashboard.exports import export_dataframe_to_excel, export_dataframe_to_pdf


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Vehicle": ["EVR-0001", "EVR-0002"],
        "Risk": ["High", "Minimal"],
        "Charging Stress": ["82%", "10%"],
    })


class TestExportToExcel:
    def test_round_trips_via_pandas(self, sample_df):
        raw = export_dataframe_to_excel(sample_df)
        result = pd.read_excel(BytesIO(raw))
        pd.testing.assert_frame_equal(result, sample_df)

    def test_sheet_name_is_respected(self, sample_df):
        raw = export_dataframe_to_excel(sample_df, sheet_name="MySheet")
        sheets = pd.ExcelFile(BytesIO(raw)).sheet_names
        assert sheets == ["MySheet"]

    def test_empty_dataframe_still_produces_valid_file(self):
        raw = export_dataframe_to_excel(pd.DataFrame(columns=["A", "B"]))
        result = pd.read_excel(BytesIO(raw))
        assert list(result.columns) == ["A", "B"]
        assert result.empty


class TestExportToPdf:
    def test_produces_a_valid_pdf_byte_stream(self, sample_df):
        raw = export_dataframe_to_pdf(sample_df, title="Test Report")
        assert raw.startswith(b"%PDF")
        assert raw.rstrip().endswith(b"%%EOF")

    def test_nonempty_for_larger_table(self):
        df = pd.DataFrame({f"col_{i}": range(30) for i in range(8)})
        raw = export_dataframe_to_pdf(df, title="Large Report")
        assert len(raw) > 0
        assert raw.startswith(b"%PDF")

    def test_empty_dataframe_does_not_raise(self):
        raw = export_dataframe_to_pdf(pd.DataFrame(columns=["A", "B"]), title="Empty")
        assert raw.startswith(b"%PDF")
