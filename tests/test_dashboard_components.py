"""Tests for UI component accessibility attributes."""
from __future__ import annotations

from src.dashboard.components.alert_badge import create_status_indicator
from src.dashboard.components.metric_card import create_metric_card


def test_status_indicator_accessibility():
    indicator = create_status_indicator("healthy")
    assert indicator.role == "img"
    assert indicator.title == "Status: healthy"
    assert getattr(indicator, "aria-label") == "Status: healthy"


def test_metric_card_accessibility():
    card = create_metric_card("total-predictions", "Total Predictions", "API")
    card_body = card.children
    icon_div = card_body.children[0]

    assert icon_div.role == "img"
    assert icon_div.title == "Total Predictions icon"
    assert getattr(icon_div, "aria-label") == "Total Predictions icon"

    status_span = card_body.children[3].children[1]
    assert status_span.role == "img"
    assert status_span.title == "Metric status"
    assert getattr(status_span, "aria-label") == "Metric status"
