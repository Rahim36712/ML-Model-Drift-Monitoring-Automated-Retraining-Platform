"""Top navigation bar for the MLOps Drift Monitor dashboard."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def create_navbar() -> dbc.Navbar:
    """Return the responsive top navigation bar."""

    nav_items = [
        dbc.NavItem(
            dbc.NavLink(
                label,
                href=href,
                className="nav-link-custom",
                active="exact",
            )
        )
        for label, href in [
            ("Overview", "/overview"),
            ("Data Drift", "/data-drift"),
            ("Prediction Drift", "/prediction-drift"),
            ("Performance", "/performance"),
            ("Alerts", "/alerts"),
            ("Model Registry", "/model-registry"),
        ]
    ]

    status_cluster = dbc.Nav(
        [
            dbc.NavItem(
                html.Span(
                    [
                        html.Span(
                            id="health-badge",
                            className="status-dot status-dot-gray",
                        ),
                        dbc.Badge(
                            "Healthy",
                            id="health-badge-text",
                            color="success",
                            pill=True,
                            className="me-3",
                            style={"fontSize": "0.75rem"},
                        ),
                    ],
                    className="d-flex align-items-center",
                ),
            ),
            dbc.NavItem(
                html.Small(
                    [
                        html.Span(
                            "Last updated: ",
                            style={"color": "var(--color-text-muted)"},
                        ),
                        html.Span(
                            "-",
                            id="last-updated",
                            style={"color": "var(--color-text)"},
                        ),
                    ],
                    className="d-flex align-items-center",
                ),
            ),
        ],
        className="ms-auto d-flex align-items-center",
        navbar=True,
    )

    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    "MLOps Drift Monitor",
                    className="brand-gradient me-4",
                    href="/overview",
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(nav_items, className="me-auto", navbar=True),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
                status_cluster,
            ],
            fluid=True,
        ),
        className="navbar-dark-custom",
        dark=True,
        sticky="top",
        style={"padding": "0.5rem 0"},
    )
