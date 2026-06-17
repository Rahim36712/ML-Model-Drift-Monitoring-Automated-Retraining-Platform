"""Plotly Dash monitoring dashboard application factory.

Creates and configures the main Dash application with dark-themed
glassmorphism UI, multi-page routing, and real-time data refresh.
Runs on port 8050, separate from the FastAPI backend (port 8000).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from src.config.settings import get_settings
from src.dashboard.components.navbar import create_navbar

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom index HTML with dark glassmorphism CSS
# ---------------------------------------------------------------------------
_CUSTOM_INDEX = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>MLOps Drift Monitor</title>
    {%favicon%}
    {%css%}
    <style>
        /* ── Base palette ─────────────────────────────────── */
        :root {
            --color-primary:    #6366f1; /* Indigo */
            --color-success:    #10b981; /* Emerald green */
            --color-warning:    #f59e0b; /* Amber */
            --color-danger:     #ef4444; /* Rose/Red */
            --color-text:       #f3f4f6; /* Slate 100 */
            --color-text-muted: #94a3b8; /* Slate 400 */
            --color-bg:         #030712; /* Deep space obsidian/gray 950 */
            --color-card-bg:    rgba(17, 24, 39, 0.70); /* Glassy dark slate 900 */
            --color-border:     rgba(255, 255, 255, 0.06);
            --gradient-accent:  linear-gradient(135deg, #a855f7 0%, #6366f1 100%); /* Purple to Indigo */
            --font-family:      'Inter', -apple-system, BlinkMacSystemFont,
                                'Segoe UI', Roboto, sans-serif;
        }

        /* ── Global ───────────────────────────────────────── */
        * { box-sizing: border-box; }

        body {
            font-family: var(--font-family) !important;
            background: var(--color-bg) !important;
            color: var(--color-text) !important;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* ── Glassmorphism card ────────────────────────────── */
        .glass-card {
            background: var(--color-card-bg) !important;
            backdrop-filter: blur(18px) saturate(180%);
            -webkit-backdrop-filter: blur(18px) saturate(180%);
            border: 1px solid var(--color-border) !important;
            border-left: 4px solid transparent !important;
            border-image: var(--gradient-accent) 1 !important;
            border-image-slice: 1 !important;
            border-radius: 16px !important; /* Larger, sleeker round corners */
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), 
                        box-shadow 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                        border-color 0.25s ease;
        }

        /* Fix border-radius when using border-image on glass-card */
        /* To preserve rounded corners with border gradients, we apply overflow: hidden or style custom clips.
           Since border-image slices can sometimes disable border-radius in some browsers, we can use a box shadow or 
           traditional background clip. Let's make sure the corners are visibly rounded by setting overflow and border radius. */
        .glass-card {
            overflow: visible;
        }

        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 25px rgba(99, 102, 241, 0.20),
                        0 10px 30px rgba(0, 0, 0, 0.50) !important;
            border-color: rgba(99, 102, 241, 0.40) !important;
        }

        /* ── Metric card specifics ─────────────────────────── */
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--color-text);
            line-height: 1.1;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #e2e8f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-delta-up {
            color: var(--color-success);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-delta-down {
            color: var(--color-danger);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-title {
            color: var(--color-text-muted);
            font-size: 0.80rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        .metric-icon {
            font-size: 1.6rem;
            margin-bottom: 0.25rem;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }

        /* ── Status dots ──────────────────────────────────── */
        .status-dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .status-dot-green   { background: var(--color-success); box-shadow: 0 0 8px var(--color-success); }
        .status-dot-yellow  { background: var(--color-warning); box-shadow: 0 0 8px var(--color-warning); }
        .status-dot-red     { background: var(--color-danger);  box-shadow: 0 0 8px var(--color-danger);  }
        .status-dot-gray    { background: #64748b; }

        /* ── Chart containers ─────────────────────────────── */
        .chart-card .card-header {
            background: transparent !important;
            border-bottom: 1px solid var(--color-border) !important;
            color: var(--color-text) !important;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            padding: 1rem 1.25rem !important;
        }

        .chart-card .card-body {
            padding: 0.75rem !important;
        }

        /* ── Navbar overrides ─────────────────────────────── */
        .navbar-dark-custom {
            background: rgba(3, 7, 18, 0.90) !important;
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--color-border);
            padding: 0.75rem 2rem !important;
        }

        .brand-gradient {
            background: linear-gradient(135deg, #a855f7, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: -0.01em;
        }

        .nav-link-custom {
            color: var(--color-text-muted) !important;
            font-weight: 500;
            transition: color 0.25s ease, transform 0.25s ease;
            font-size: 0.9rem;
            padding: 0.5rem 1rem !important;
            border-radius: 8px;
        }

        .nav-link-custom:hover,
        .nav-link-custom.active {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.05);
        }

        /* ── Scrollbar ────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--color-bg); }
        ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #374151; }

        /* ── Page content container ────────────────────────── */
        #page-content {
            padding: 2rem;
            max-width: 1600px;
            margin: 0 auto;
        }

        /* ── Dropdown Overrides ────────────────────────────── */
        .Select-control, .dash-dropdown .Select-control {
            background-color: #111827 !important; /* slate 900 */
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            color: #f9fafb !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
            height: 38px !important;
        }
        .Select-control:hover, .dash-dropdown .Select-control:hover {
            border-color: #6366f1 !important;
        }
        .Select-value-label, .Select-placeholder, .Select-input > input,
        .dash-dropdown .Select-value-label, .dash-dropdown .Select-placeholder {
            color: #f3f4f6 !important;
            line-height: 36px !important;
        }
        .Select-menu-outer, .dash-dropdown .Select-menu-outer {
            background-color: #111827 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.70) !important;
            z-index: 9999 !important; /* Ensure dropdown displays above chart boxes */
        }
        .Select-option, .dash-dropdown .Select-option {
            background-color: transparent !important;
            color: #94a3b8 !important;
            padding: 10px 14px !important;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .Select-option:hover, .Select-option.is-focused,
        .dash-dropdown .Select-option:hover, .dash-dropdown .Select-option.is-focused {
            background-color: #4f46e5 !important;
            color: #ffffff !important;
        }
        .Select-option.is-selected, .dash-dropdown .Select-option.is-selected {
            background-color: rgba(99, 102, 241, 0.25) !important;
            color: #818cf8 !important;
        }
        .Select-clear-zone, .Select-arrow-zone {
            padding-top: 4px;
        }
        .Select-arrow {
            border-color: #94a3b8 transparent transparent !important;
        }
        .is-open > .Select-control .Select-arrow {
            border-color: transparent transparent #94a3b8 !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""


# Clean, ASCII-safe monitoring console theme. This overrides the earlier
# generated theme block and avoids mojibake in visible UI.
_CUSTOM_INDEX = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>MLOps Drift Monitor</title>
    {%favicon%}
    {%css%}
    <style>
        :root {
            --color-primary: #22d3ee;
            --color-secondary: #a3e635;
            --color-success: #22c55e;
            --color-warning: #f59e0b;
            --color-danger: #f43f5e;
            --color-text: #e6edf3;
            --color-text-muted: #91a4b7;
            --color-bg: #071014;
            --color-panel: rgba(13, 25, 31, 0.92);
            --color-panel-strong: rgba(16, 34, 42, 0.96);
            --color-border: rgba(148, 163, 184, 0.18);
            --color-border-active: rgba(34, 211, 238, 0.45);
            --shadow-panel: 0 18px 48px rgba(0, 0, 0, 0.32);
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        * { box-sizing: border-box; }

        body {
            font-family: var(--font-family) !important;
            background:
                linear-gradient(145deg, rgba(34, 211, 238, 0.08), transparent 32%),
                linear-gradient(315deg, rgba(163, 230, 53, 0.07), transparent 26%),
                var(--color-bg) !important;
            color: var(--color-text) !important;
            margin: 0;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .glass-card {
            background: var(--color-panel) !important;
            border: 1px solid var(--color-border) !important;
            border-radius: 8px !important;
            box-shadow: var(--shadow-panel);
            transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        }

        .glass-card:hover {
            transform: translateY(-1px);
            border-color: var(--color-border-active) !important;
            box-shadow: 0 20px 54px rgba(0, 0, 0, 0.40);
        }

        .metric-value {
            color: var(--color-text);
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .metric-title {
            color: var(--color-text-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .metric-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2.1rem;
            height: 2.1rem;
            padding: 0 0.55rem;
            margin-bottom: 0.75rem;
            border-radius: 8px;
            color: var(--color-primary);
            background: rgba(34, 211, 238, 0.12);
            border: 1px solid rgba(34, 211, 238, 0.24);
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0;
        }

        .metric-delta-up,
        .metric-delta-down {
            color: var(--color-text-muted);
            font-size: 0.85rem;
            font-weight: 650;
        }

        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .status-dot-green { background: var(--color-success); box-shadow: 0 0 10px rgba(34, 197, 94, 0.7); }
        .status-dot-yellow { background: var(--color-warning); box-shadow: 0 0 10px rgba(245, 158, 11, 0.7); }
        .status-dot-red { background: var(--color-danger); box-shadow: 0 0 10px rgba(244, 63, 94, 0.7); }
        .status-dot-gray { background: #64748b; }

        .chart-card .card-header {
            background: var(--color-panel-strong) !important;
            border-bottom: 1px solid var(--color-border) !important;
            color: var(--color-text) !important;
            font-weight: 700;
            font-size: 0.92rem;
            padding: 0.9rem 1rem !important;
        }

        .chart-card .card-body {
            padding: 0.75rem !important;
        }

        .navbar-dark-custom {
            background: rgba(7, 16, 20, 0.94) !important;
            backdrop-filter: blur(14px);
            border-bottom: 1px solid var(--color-border);
            padding: 0.65rem 1.25rem !important;
        }

        .brand-gradient {
            color: var(--color-text) !important;
            font-weight: 820;
            font-size: 1.05rem;
            letter-spacing: 0;
        }

        .brand-gradient::before {
            content: "";
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 0.6rem;
            border-radius: 3px;
            background: var(--color-primary);
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.85);
        }

        .nav-link-custom {
            color: var(--color-text-muted) !important;
            font-size: 0.88rem;
            font-weight: 650;
            padding: 0.45rem 0.75rem !important;
            border-radius: 8px;
            transition: background 0.18s ease, color 0.18s ease;
        }

        .nav-link-custom:hover,
        .nav-link-custom.active {
            color: var(--color-text) !important;
            background: rgba(34, 211, 238, 0.12);
        }

        #page-content {
            padding: 1.5rem;
            max-width: 1560px;
            margin: 0 auto;
        }

        .Select-control, .dash-dropdown .Select-control {
            background-color: #10222a !important;
            border: 1px solid var(--color-border) !important;
            border-radius: 8px !important;
            color: var(--color-text) !important;
            height: 38px !important;
        }
        .Select-control:hover, .dash-dropdown .Select-control:hover {
            border-color: var(--color-border-active) !important;
        }
        .Select-value-label, .Select-placeholder, .Select-input > input,
        .dash-dropdown .Select-value-label, .dash-dropdown .Select-placeholder {
            color: var(--color-text) !important;
            line-height: 36px !important;
        }
        .Select-menu-outer, .dash-dropdown .Select-menu-outer {
            background-color: #10222a !important;
            border: 1px solid var(--color-border-active) !important;
            border-radius: 8px !important;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.55) !important;
            z-index: 9999 !important;
        }
        .Select-option, .dash-dropdown .Select-option {
            color: var(--color-text-muted) !important;
            background-color: transparent !important;
            padding: 10px 14px !important;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .Select-option:hover, .Select-option.is-focused,
        .dash-dropdown .Select-option:hover, .dash-dropdown .Select-option.is-focused {
            background-color: rgba(34, 211, 238, 0.14) !important;
            color: var(--color-text) !important;
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--color-bg); }
        ::-webkit-scrollbar-thumb { background: #25424c; border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: #315865; }

        @media (max-width: 768px) {
            #page-content { padding: 1rem; }
            .navbar-dark-custom { padding: 0.65rem 0.75rem !important; }
            .metric-value { font-size: 1.7rem; }
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_dash_app() -> dash.Dash:
    """Build and return a fully-configured Dash application.

    The app uses the DARKLY Bootstrap theme, custom dark CSS with
    glassmorphism effects, and multi-page routing via ``dcc.Location``.
    """
    settings = get_settings()

    # External stylesheets – Bootstrap DARKLY + Google Fonts (Inter)
    external_stylesheets = [
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ]

    app = dash.Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
        title="MLOps Drift Monitor",
        update_title="Updating… | MLOps Drift Monitor",
    )

    # Inject custom index HTML with embedded CSS
    app.index_string = _CUSTOM_INDEX

    # Refresh interval (ms) – falls back to 5 000 ms
    refresh_ms: int = getattr(
        getattr(settings, "dashboard", None), "refresh_interval_ms", 5000
    )

    # ── Root layout ──────────────────────────────────────────────────────
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Interval(
                id="interval-refresh",
                interval=refresh_ms,
                n_intervals=0,
            ),
            create_navbar(),
            html.Div(id="page-content"),
        ]
    )

    # ── Register callbacks (import here to avoid circular imports) ──────
    _register_routing(app)
    _register_callbacks(app)

    logger.info(
        "Dash app created  ·  refresh=%dms  ·  theme=DARKLY",
        refresh_ms,
    )
    return app


# ---------------------------------------------------------------------------
# Routing callback
# ---------------------------------------------------------------------------
def _register_routing(app: dash.Dash) -> None:
    """Map URL paths to page layout functions."""
    # Lazy imports so layout modules can reference the app without cycles.
    from src.dashboard.layouts.alerts import layout as alerts_layout
    from src.dashboard.layouts.data_drift import layout as data_drift_layout
    from src.dashboard.layouts.model_registry import layout as model_registry_layout
    from src.dashboard.layouts.overview import layout as overview_layout
    from src.dashboard.layouts.performance import layout as performance_layout
    from src.dashboard.layouts.prediction_drift import layout as prediction_drift_layout

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def _route(pathname: str | None) -> html.Div:
        """Return the appropriate page layout for *pathname*."""
        routes: dict[str, callable] = {
            "/": overview_layout,
            "/overview": overview_layout,
            "/data-drift": data_drift_layout,
            "/prediction-drift": prediction_drift_layout,
            "/performance": performance_layout,
            "/alerts": alerts_layout,
            "/model-registry": model_registry_layout,
        }

        layout_fn = routes.get(pathname or "/", overview_layout)
        try:
            return layout_fn()
        except Exception:
            logger.exception("Failed to render layout for %s", pathname)
            return html.Div(
                [
                    html.H3("Page Error", className="text-warning mt-4"),
                    html.P(
                        f"Could not load page: {pathname}",
                        className="text-muted",
                    ),
                ],
                className="text-center",
            )


def _register_callbacks(app: dash.Dash) -> None:
    """Import and register all dashboard callback modules."""
    from src.dashboard.callbacks import (  # noqa: F401 – side-effect imports
        overview_callbacks,
        drift_callbacks,
        performance_callbacks,
    )

    @app.callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("navbar-collapse", "is_open"),
    )
    def _toggle_navbar(n_clicks: int | None, is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open

    logger.info("Registered dashboard callbacks successfully.")


# ---------------------------------------------------------------------------
# Entrypoint helpers
# ---------------------------------------------------------------------------
def run_dashboard() -> None:
    """Load settings, create the Dash app, and start the dev server."""
    settings = get_settings()

    host: str = getattr(
        getattr(settings, "dashboard", None), "host", "0.0.0.0"
    )
    port: int = getattr(
        getattr(settings, "dashboard", None), "port", 8050
    )

    app = create_dash_app()
    logger.info("Starting dashboard on %s:%d", host, port)
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run_dashboard()
