"""
tests/test_bi_chat_dashboard.py

Validates dashboard/components/bi_chat.py's pure, non-streamlit chart
builders -- specifically build_chart_from_spec, which turns the BI
chat agent's dynamic chart spec into an Altair chart. Doesn't touch
st.* at all, so no running Streamlit app/session is needed, matching
the existing pattern in tests for dashboard/components/fleet_map.py's
build_* functions.

Run from the project root:
    pytest tests/test_bi_chat_dashboard.py -v
"""

from dashboard.components.bi_chat import build_chart_from_spec


class TestBuildChartFromSpec:
    def test_valid_bar_spec_builds_a_chart(self):
        spec = {
            "type": "bar", "title": "t", "x_field": "vehicle_id", "y_field": "score",
            "data": [{"vehicle_id": "EVR-0001", "score": 10}, {"vehicle_id": "EVR-0002", "score": 20}],
        }
        chart = build_chart_from_spec(spec)
        assert chart is not None

    def test_valid_line_spec_with_series_builds_a_chart(self):
        spec = {
            "type": "line", "title": "t", "x_field": "cycle", "y_field": "value",
            "series_field": "vehicle_id",
            "data": [
                {"cycle": 1, "value": 99.0, "vehicle_id": "EVR-0001"},
                {"cycle": 2, "value": 98.5, "vehicle_id": "EVR-0001"},
            ],
        }
        chart = build_chart_from_spec(spec)
        assert chart is not None

    def test_scatter_spec_builds_a_chart(self):
        spec = {
            "type": "scatter", "title": "t", "x_field": "charge_stress_score", "y_field": "thermal_anomaly_count",
            "data": [{"charge_stress_score": 10, "thermal_anomaly_count": 1}],
        }
        chart = build_chart_from_spec(spec)
        assert chart is not None

    def test_empty_data_returns_none(self):
        spec = {"type": "bar", "x_field": "a", "y_field": "b", "data": []}
        assert build_chart_from_spec(spec) is None

    def test_missing_x_field_in_data_returns_none(self):
        spec = {"type": "bar", "x_field": "not_present", "y_field": "score", "data": [{"score": 1}]}
        assert build_chart_from_spec(spec) is None

    def test_table_type_returns_none_caller_renders_dataframe_instead(self):
        spec = {"type": "table", "x_field": "a", "y_field": "b", "data": [{"a": 1, "b": 2}]}
        assert build_chart_from_spec(spec) is None

    def test_unknown_type_returns_none(self):
        spec = {"type": "pie", "x_field": "a", "y_field": "b", "data": [{"a": 1, "b": 2}]}
        assert build_chart_from_spec(spec) is None