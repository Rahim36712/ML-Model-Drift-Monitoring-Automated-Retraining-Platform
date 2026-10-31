"""DriftWatch: an API-backed, accessible light monitoring workspace."""

import logging
from pathlib import Path
import requests
import dash
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from flask import redirect
from src.config.settings import get_settings
from src.runtime import api_url, management_token
from src.dashboard import views as view

NAVIGATION = [
    ("Overview", "/overview", "◫"),
    ("Drift Analysis", "/drift", "≋"),
    ("Predictions", "/predictions", "↗"),
    ("Models & Retraining", "/models", "◇"),
    ("Alerts", "/alerts", "♧"),
    ("Settings", "/settings", "⚙"),
]
ALIASES = {
    "/": "/overview",
    "/data-drift": "/drift",
    "/prediction-drift": "/drift",
    "/performance": "/drift",
    "/model-registry": "/models",
}


def api(method, path, **kwargs):
    response = requests.request(
        method,
        api_url() + path,
        headers={"X-MLOps-Token": management_token()},
        timeout=60,
        **kwargs,
    )
    if not response.ok:
        try:
            message = response.json().get("detail", "Request failed")
        except ValueError:
            message = "Backend returned an invalid response"
        raise RuntimeError(f"{message} (HTTP {response.status_code})")
    return response.json()


def create_dash_app():
    settings = get_settings()
    app = dash.Dash(
        __name__,
        assets_folder=str(Path(__file__).parent / "assets"),
        suppress_callback_exceptions=True,
        title="DriftWatch · Model Monitoring",
        update_title=None,
    )
    app.index_string = """<!DOCTYPE html><html lang="en"><head>{%metas%}<title>{%title%}</title><link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">{%css%}</head>
+<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>""".replace(
        "\n+", "\n"
    )
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="snapshot"),
            dcc.Store(id="action-message"),
            dcc.Interval(
                id="interval-refresh",
                interval=settings.dashboard.refresh_interval_ms,
                n_intervals=0,
            ),
            dcc.Download(id="csv-download"),
            dcc.ConfirmDialog(
                id="reset-dialog",
                message="Clear generated demo predictions? Models, source data, user predictions, and audit history will be kept.",
            ),
            html.Aside(
                [
                    dcc.Link(
                        [html.Span("≋", className="brand-mark"), "DriftWatch"],
                        href="/overview",
                        className="brand",
                    ),
                    html.Div(
                        [
                            html.Span("F", className="workspace-icon"),
                            html.Div(
                                [
                                    html.Strong("Fraud detection"),
                                    html.Div("Portfolio workspace", className="muted"),
                                ]
                            ),
                        ],
                        className="workspace-label",
                    ),
                    html.Div("WORKSPACE", className="eyebrow"),
                    html.Nav(
                        id="navigation", className="nav-group", **{"aria-label": "Main navigation"}
                    ),
                    html.Div(
                        [
                            view.badge("LOCAL WORKSPACE"),
                            html.Strong("Built for a changing world."),
                            "Monitor. Learn. Recover.",
                            html.Br(),
                            "Single serving worker · v2.0",
                        ],
                        className="sidebar-bottom",
                    ),
                ],
                className="sidebar",
            ),
            html.Main(
                [
                    html.Header(
                        [
                            html.Div(
                                ["Workspace", html.Strong("/  Fraud monitoring")],
                                className="topbar-left",
                            ),
                            html.Div(id="service-status", className="topbar-right"),
                        ],
                        className="topbar",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Model", htmlFor="model-version", className="filter-label"
                                    ),
                                    dcc.Dropdown(
                                        id="model-version",
                                        options=[{"label": "All versions", "value": "all"}],
                                        value="all",
                                        clearable=False,
                                        className="dropdown",
                                    ),
                                    html.Label(
                                        "Period", htmlFor="time-window", className="filter-label"
                                    ),
                                    dcc.Dropdown(
                                        id="time-window",
                                        options=[
                                            {"label": "Last hour", "value": 1},
                                            {"label": "Last 24 hours", "value": 24},
                                            {"label": "Last 7 days", "value": 168},
                                        ],
                                        value=24,
                                        clearable=False,
                                        className="dropdown",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="auto-refresh",
                                                options=[{"label": "Auto-refresh", "value": "on"}],
                                                value=["on"],
                                            ),
                                            html.Span(id="refresh-time"),
                                        ],
                                        className="filter-end",
                                    ),
                                ],
                                className="filters",
                            ),
                            dcc.Loading(html.Div(id="action-notice"), type="dot", color=view.TEAL),
                            html.Div(id="page-content"),
                            html.Footer(
                                [
                                    html.Span("DRIFTWATCH / MODEL OBSERVABILITY"),
                                    html.Span("Measured signals. Traceable decisions."),
                                ],
                                className="footer",
                            ),
                        ],
                        className="content-wrap",
                    ),
                ],
                className="main",
            ),
        ],
        className="app-shell",
    )

    for old, new in ALIASES.items():
        if old != "/":
            app.server.add_url_rule(old, "legacy-" + old, lambda target=new: redirect(target))

    @app.callback(Output("interval-refresh", "disabled"), Input("auto-refresh", "value"))
    def pause(value):
        return "on" not in (value or [])

    @app.callback(
        Output("snapshot", "data"),
        Output("model-version", "options"),
        Input("interval-refresh", "n_intervals"),
        Input("model-version", "value"),
        Input("time-window", "value"),
        Input("action-message", "data"),
    )
    def poll(_n, model, hours, _message):
        try:
            data = api(
                "GET",
                "/dashboard/snapshot",
                params={"hours": hours, **({"model_version": model} if model != "all" else {})},
            )
            options = [{"label": "All versions", "value": "all"}] + [
                {"label": "Version " + str(m["version"]), "value": str(m["version"])}
                for m in data["models"]
            ]
            return data, options
        except (requests.RequestException, RuntimeError) as exc:
            return {"error": str(exc)}, no_update

    @app.callback(Output("navigation", "children"), Input("url", "pathname"))
    def navigation(path):
        path = ALIASES.get(path, path)
        return [
            dcc.Link(
                [html.Span(icon, className="nav-icon"), label],
                href=url,
                className="nav-link" + (" active" if url == path else ""),
                title=label + (" (current page)" if url == path else ""),
            )
            for label, url, icon in NAVIGATION
        ]

    @app.callback(
        Output("page-content", "children"),
        Output("service-status", "children"),
        Output("refresh-time", "children"),
        Input("snapshot", "data"),
        Input("url", "pathname"),
        Input({"type": "view-filter", "name": ALL}, "value"),
        State({"type": "view-filter", "name": ALL}, "id"),
    )
    def render(data, path, values, ids):
        if not data:
            return (
                view.empty("Connecting to your workspace…", "The API and worker are starting."),
                view.badge("Connecting", "blue"),
                "",
            )
        if data.get("error"):
            return (
                html.Div(
                    [
                        html.H2("Backend unavailable"),
                        html.P(data["error"]),
                        html.P(
                            "Start the platform with start-demo.ps1 or python -m scripts.run_demo. Inspect logs if it is already running."
                        ),
                    ],
                    className="notice error",
                ),
                view.badge("Offline", "red"),
                "",
            )
        selected = {item["name"]: value for item, value in zip(ids, values)}
        path = ALIASES.get(path, path)
        if path == "/overview":
            content = view.overview(data)
        elif path == "/drift":
            content = view.drift_view(data, selected.get("feature", "all"))
        elif path == "/predictions":
            content = view.predictions_view(data)
        elif path == "/models":
            content = view.models_view(data)
        elif path == "/alerts":
            content = view.alerts_view(data, selected.get("search", ""))
        elif path == "/settings":
            content = view.settings_view(data)
        else:
            content = [
                view.heading("Page not found", "Choose a page from the navigation."),
                dcc.Link("Return to Overview", href="/overview"),
            ]
        drift = data.get("drift") or {}
        if drift.get("stale"):
            content.insert(
                1,
                html.Div(
                    "The last drift result is stale or belongs to another model. Generate traffic and run a new check.",
                    className="notice warning",
                ),
            )
        status = [
            view.badge(
                "API ready" if data["ready"] else "Model unavailable",
                "teal" if data["ready"] else "red",
            ),
            view.badge(data["source"].upper() + " DATA", "blue"),
            html.Span("v" + str(data["active_version"] or "—"), className="muted"),
        ]
        return content, status, "Updated " + view.stamp(data["timestamp"])[11:]

    @app.callback(
        Output("action-message", "data"),
        Output("reset-dialog", "displayed"),
        Output("url", "pathname"),
        Output("csv-download", "data"),
        Input({"type": "command", "action": ALL}, "n_clicks"),
        Input("reset-dialog", "submit_n_clicks"),
        State("snapshot", "data"),
        prevent_initial_call=True,
    )
    def action(clicks, reset_clicks, data):
        trigger = ctx.triggered_id
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update, no_update, no_update
        if trigger != "reset-dialog" and (
            not isinstance(trigger, dict)
            or not isinstance(trigger.get("action"), str)
            or not any(clicks or [])
        ):
            return no_update, no_update, no_update, no_update
        try:
            if trigger == "reset-dialog":
                result = api("POST", "/demo/reset", json={"confirmation": "RESET DEMO"})
            else:
                action_name = trigger["action"]
                if action_name == "reset":
                    return no_update, True, no_update, no_update
                if action_name == "results":
                    return no_update, False, "/models", no_update
                if action_name == "export":
                    import pandas as pd

                    rows = [
                        {k: v for k, v in row.items() if k != "features_json"}
                        for row in data.get("predictions", [])
                    ]
                    return (
                        no_update,
                        False,
                        no_update,
                        dcc.send_data_frame(
                            pd.DataFrame(rows).to_csv, "driftwatch-predictions.csv", index=False
                        ),
                    )
                if action_name.startswith("traffic-"):
                    result = api(
                        "POST",
                        "/demo/traffic",
                        json={"scenario": action_name.split("-")[1], "samples": 500},
                    )
                elif action_name == "check":
                    result = api("POST", "/drift/run")
                elif action_name == "retrain":
                    result = api("POST", "/models/retrain")
                elif action_name.startswith("rollback-"):
                    result = api("POST", "/models/rollback/" + action_name.split("-")[1])
                elif action_name.startswith("ack-"):
                    result = api("POST", "/alerts/" + action_name.split("-")[1] + "/acknowledge")
                else:
                    return no_update, no_update, no_update, no_update
            message = result.get("message") or (
                "Job submitted: " + result["job_id"][:8]
                if result.get("job_id")
                else "Drift check complete: " + result["overall_status"]
                if result.get("overall_status")
                else "Action completed successfully."
            )
            return {"text": message, "error": False}, False, no_update, no_update
        except (requests.RequestException, RuntimeError) as exc:
            return {"text": str(exc), "error": True}, False, no_update, no_update

    @app.callback(Output("action-notice", "children"), Input("action-message", "data"))
    def notice(message):
        return (
            html.Div(
                message["text"], className="notice error" if message.get("error") else "notice"
            )
            if message
            else None
        )

    return app


def run_dashboard():
    settings = get_settings()
    app = create_dash_app()
    app.run(host=settings.dashboard.host, port=settings.dashboard.port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()
