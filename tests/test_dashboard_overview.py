"""The visible GUI uses real measurements and the newest monitoring history."""

from datetime import datetime, timedelta
from src.dashboard.views import trend
from src.dashboard.app import create_dash_app


def test_overview_shows_latest_100_measurements_in_time_order():
    start = datetime(2026, 1, 1)
    records = [
        {
            "timestamp": (start + timedelta(minutes=i)).isoformat(),
            "drift_type": "data",
            "metric_name": "psi",
            "metric_value": i / 1000,
        }
        for i in reversed(range(105))
    ]
    component = trend(
        {
            "drifts": records,
            "thresholds": {"data_drift": {"psi": {"warning": 0.1, "critical": 0.25}}},
        },
        "data",
    )
    trace = component.figure.data[0]
    assert len(trace.x) == 100
    assert trace.x[0] == (start + timedelta(minutes=5)).isoformat()
    assert trace.x[-1] == (start + timedelta(minutes=104)).isoformat()
    assert list(trace.y) == [i / 1000 for i in range(5, 105)]


def test_dashboard_layout_and_local_assets(runtime):
    app = create_dash_app()
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/data-drift").status_code == 302


def test_every_empty_page_serializes(runtime):
    import json
    from plotly.utils import PlotlyJSONEncoder
    from src.dashboard import views
    from src.service import dashboard_snapshot
    from src.runtime import management_token

    data = dashboard_snapshot(runtime[1], runtime[0])
    pages = [
        views.overview(data),
        views.drift_view(data, "all"),
        views.predictions_view(data),
        views.models_view(data),
        views.alerts_view(data, ""),
        views.settings_view(data),
    ]
    encoded = json.dumps(pages, cls=PlotlyJSONEncoder, default=PlotlyJSONEncoder().default)
    assert management_token() not in encoded
    assert "Generate 500 predictions" in encoded
