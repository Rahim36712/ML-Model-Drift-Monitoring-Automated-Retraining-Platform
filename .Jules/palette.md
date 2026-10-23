# Palette's Journal - MLOps Drift Monitor

## 2026-03-31 - Dash Bootstrap Components ARIA Attributes Constraint
**Learning:** `dash_bootstrap_components` elements (e.g. `dbc.Badge`, `dbc.Card`, `dbc.NavbarToggler`) and `dcc` components strictly validate keyword arguments against their predefined schema in Dash. Passing `aria-label` or `role` directly to `dbc` or `dcc` components raises a `TypeError`. However, standard `dash.html` components (`html.Span`, `html.Div`, etc.) support arbitrary ARIA attributes such as `aria-label`, `role`, and `title`.
**Action:** When adding accessibility features in Plotly Dash apps, attach `aria-label` and `role` to standard `html` elements or wrap `dbc`/`dcc` components in `html.Div`/`html.Span` wrappers when needed.
