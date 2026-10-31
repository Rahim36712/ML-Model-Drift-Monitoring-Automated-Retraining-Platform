"""Light dashboard views. All displayed measurements come from API snapshots."""

import json
from dash import dcc, html
import plotly.graph_objects as go

TEAL = "#087f82"
MUTED = "#607188"


def badge(label, color="teal"):
    return html.Span([html.Span(className="badge-dot"), str(label)], className=f"badge {color}")


def command(label, action, primary=False, disabled=False, extra=""):
    return html.Button(
        label,
        id={"type": "command", "action": action},
        n_clicks=0,
        disabled=disabled,
        className="btn " + ("btn-primary " if primary else "") + extra,
    )


def card(title, children, subtitle=None):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(title),
                            html.P(subtitle, className="card-subtitle") if subtitle else None,
                        ]
                    )
                ],
                className="card-head",
            ),
            children,
        ],
        className="card",
    )


def empty(title, hint="Generate traffic to start collecting measurements."):
    return html.Div([html.Strong(title), hint], className="empty")


def table(headers, rows):
    if not rows:
        return empty("Nothing here yet", "Records will appear as you use the platform.")
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    value,
                                    className="message"
                                    if isinstance(value, str) and len(value) > 50
                                    else "",
                                )
                                for value in row
                            ]
                        )
                        for row in rows
                    ]
                ),
            ]
        ),
        className="table-wrap",
    )


def metric(title, value, note, icon):
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(icon, className="metric-icon"),
            html.Div(value, className="metric-value"),
            html.Div(note, className="metric-note"),
        ],
        className="card metric",
    )


def chart(figure, chart_id):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial", color=MUTED, size=11),
        margin=dict(l=48, r=25, t=25, b=42),
        height=290,
        hovermode="x unified",
        uirevision=chart_id,
        xaxis=dict(gridcolor="#edf1f7", zeroline=False),
        yaxis=dict(gridcolor="#edf1f7", zeroline=False),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    if not figure.data:
        figure.add_annotation(
            text="No measurements available yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color=MUTED),
        )
    return dcc.Graph(
        id=chart_id, figure=figure, config={"displayModeBar": False, "responsive": True}
    )


def number(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "—"


def stamp(value):
    if not value:
        return "Not available"
    return str(value).replace("T", " ")[:19] + " UTC"


def busy(snapshot):
    return any(
        job["status"] in ["QUEUED", "RUNNING", "EVALUATING", "DEPLOYING"]
        for job in snapshot.get("jobs", [])
    )


def heading(title, subtitle, tag="LIVE WORKSPACE"):
    return html.Div(
        [html.Div([html.H1(title), html.P(subtitle, className="subtitle")]), badge(tag, "blue")],
        className="page-heading",
    )


def trend(snapshot, kind):
    rows = [r for r in reversed(snapshot.get("drifts", [])) if r["drift_type"] == kind]
    if kind == "data":
        rows = [r for r in rows if r["metric_name"] == "psi"]
    # Concept checks can have several rows; show one point per completed window.
    rows = list({r["timestamp"]: r for r in rows}.values())[-100:]
    fig = go.Figure()
    if rows:
        values = (
            [r["metric_value"] for r in rows]
            if kind == "data"
            else [
                json.loads(r["details_json"] or "{}").get("current_metrics", {}).get("f1")
                for r in rows
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=[r["timestamp"] for r in rows],
                y=values,
                mode="lines+markers",
                line=dict(color=TEAL, width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(8,127,130,.05)",
                name="PSI" if kind == "data" else "Fraud F1",
            )
        )
    if kind == "data":
        thresholds = snapshot["thresholds"]["data_drift"]["psi"]
        for label, color in [("warning", "#cf963f"), ("critical", "#d26a73")]:
            fig.add_hline(
                y=thresholds[label],
                line_dash="dot",
                line_color=color,
                annotation_text=label.title(),
                annotation_position="top right",
            )
    return chart(fig, "trend-" + kind)


def job_panel(snapshot):
    jobs = snapshot.get("jobs", [])[:6]
    if not jobs:
        return card(
            "Activity & retraining",
            empty("Ready for your first run", "Generate baseline traffic or retrain a challenger."),
        )
    blocks = []
    for job in jobs:
        result = job.get("result", {})
        tone = (
            "red"
            if job["status"] in ["FAILED", "INTERRUPTED"]
            else "orange"
            if job["status"] == "REJECTED"
            else "teal"
        )
        title = "Traffic simulation" if job["kind"] == "traffic" else "Model training"
        blocks.append(
            html.Div(
                [
                    html.Span(className="timeline-dot"),
                    html.Div(
                        [
                            html.Div(
                                [html.Strong(title), badge(job["status"], tone)],
                                className="section-heading",
                            ),
                            html.P(result.get("message") or job["reason"][:110]),
                            html.Small(stamp(job["updated_at"])),
                            html.Div(
                                html.Div(
                                    style={
                                        "width": f"{100 * result.get('processed', 0) / max(result.get('total', 1), 1)}%"
                                    },
                                    className="progress-bar",
                                ),
                                className="progress-track",
                            )
                            if result.get("total")
                            else None,
                        ],
                        className="timeline-text",
                    ),
                ],
                className="timeline-row",
            )
        )
    return card("Activity & retraining", html.Div(blocks), "Persistent jobs · actual worker status")


def overview(snapshot):
    stats = snapshot["stats"]
    drift = snapshot.get("drift") or {}
    active = next((m for m in snapshot["models"] if m["is_production"]), None)
    f1 = (drift.get("concept_drift") or {}).get("current_metrics", {}).get("f1")
    psi = (drift.get("data_drift") or {}).get("overall_psi")
    alerts = stats["open_alerts"]
    running = busy(snapshot)
    steps = [
        (
            "01",
            "Generate traffic",
            "Real inference with source labels.",
            "Generate 500 predictions",
            "traffic-stable",
        ),
        (
            "02",
            "Introduce drift",
            "Shift five features in a seeded scenario.",
            "Simulate feature shift",
            "traffic-drift",
        ),
        (
            "03",
            "Check model health",
            "Compare recent traffic to its baseline.",
            "Run drift check",
            "check",
        ),
        (
            "04",
            "Inspect the outcome",
            "Follow quality gates and model activation.",
            "View training results",
            "results",
        ),
    ]
    demo = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Your model. A changing world."),
                            html.P(
                                "Walk through a real monitoring and recovery lifecycle in four steps.",
                                className="subtitle",
                            ),
                        ]
                    ),
                    badge("GUIDED DEMO"),
                ],
                className="demo-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(num, className="step-number"),
                            html.Div(title, className="step-title"),
                            html.P(description, className="step-description"),
                            command(
                                label,
                                action,
                                primary=index == 0,
                                disabled=running and action != "results",
                                extra="step-btn",
                            ),
                        ],
                        className="demo-step",
                    )
                    for index, (num, title, description, label, action) in enumerate(steps)
                ],
                className="demo-steps",
            ),
        ],
        className="card demo-card",
    )
    next_step = (
        "A job is running. Follow its progress below."
        if running
        else "Generate traffic to begin."
        if stats["total"] == 0
        else "Run a drift check to compare traffic with the current model."
        if not drift.get("window")
        else "Review the latest drift and training results. A rejected challenger keeps the champion active."
    )
    latency = go.Figure()
    rows = list(reversed(snapshot.get("predictions", [])))
    if rows:
        latency.add_trace(
            go.Scatter(
                x=[r["timestamp"] for r in rows],
                y=[r["latency_ms"] for r in rows],
                mode="lines",
                line=dict(color="#3d5fa9", width=2),
                name="Inference ms / record",
            )
        )
    return [
        heading(
            "Model overview", "A clear view of your model, its data, and what needs attention."
        ),
        demo,
        html.Div(
            [
                metric(
                    "Predictions",
                    f"{stats['total']:,}",
                    f"{stats['throughput_per_min']} in the last minute",
                    "↗",
                ),
                metric(
                    "Data drift · PSI",
                    number(psi),
                    "No check yet"
                    if psi is None
                    else drift.get("overall_status", "Unknown").title(),
                    "≋",
                ),
                metric(
                    "Fraud F1",
                    number(f1),
                    "Waiting for labeled monitoring data"
                    if f1 is None
                    else "Measured on labeled production traffic",
                    "◎",
                ),
                metric(
                    "Open alerts", str(alerts), "Unacknowledged warnings and critical alerts", "!"
                ),
            ],
            className="metrics",
        ),
        html.Div(
            [
                card(
                    "Feature drift over time",
                    trend(snapshot, "data"),
                    "PSI · thresholds from configuration",
                ),
                card(
                    "Inference latency",
                    chart(latency, "latency-overview"),
                    "Actual model time per record; batch cost is amortized",
                ),
            ],
            className="grid-two",
        ),
        html.Div(
            [
                job_panel(snapshot),
                card(
                    "Workspace health",
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Serving API"),
                                    badge(
                                        "Ready" if snapshot["ready"] else "Not ready",
                                        "teal" if snapshot["ready"] else "red",
                                    ),
                                ],
                                className="definition",
                            ),
                            html.Div(
                                [
                                    html.Span("Training worker"),
                                    badge(
                                        "Online" if snapshot["worker_online"] else "Offline",
                                        "teal" if snapshot["worker_online"] else "red",
                                    ),
                                ],
                                className="definition",
                            ),
                            html.Div(
                                [
                                    html.Span("Active champion"),
                                    html.Strong(
                                        f"Version {snapshot['active_version']}"
                                        if active
                                        else "Not trained"
                                    ),
                                ],
                                className="definition",
                            ),
                            html.Div(
                                [
                                    html.Span("Label coverage"),
                                    html.Strong(f"{stats['label_coverage']:.0%}"),
                                ],
                                className="definition",
                            ),
                            html.Div(
                                [
                                    html.Span("Request p95"),
                                    html.Strong(number(stats["request_p95_ms"], 2) + " ms"),
                                ],
                                className="definition",
                            ),
                            html.Div(
                                [
                                    html.Span("Latest drift check"),
                                    html.Span(stamp(drift.get("timestamp"))),
                                ],
                                className="definition",
                            ),
                            html.P(
                                next_step,
                                className="notice",
                                style={"marginTop": "18px", "marginBottom": "0"},
                            ),
                        ],
                        className="card-body",
                    ),
                ),
            ],
            className="grid-equal",
        ),
        html.P(
            "Synthetic scenarios are labeled. Drift status describes measurements, not API readiness. All displayed times are UTC.",
            className="muted",
            style={"fontSize": "11px"},
        ),
    ]


def drift_view(snapshot, feature):
    drift = snapshot.get("drift") or {}
    data = drift.get("data_drift") or {}
    psi = data.get("feature_psi", {})
    fig = go.Figure()
    if psi:
        values = sorted(psi.items(), key=lambda pair: pair[1], reverse=True)
        if feature != "all":
            values = [p for p in values if p[0] == feature]
        fig.add_trace(
            go.Bar(
                x=[p[0] for p in values],
                y=[p[1] for p in values],
                marker_color=[
                    "#c96b72"
                    if p[1] >= snapshot["thresholds"]["data_drift"]["psi"]["critical"]
                    else TEAL
                    for p in values
                ],
                name="Feature PSI",
            )
        )
        fig.add_hline(
            y=snapshot["thresholds"]["data_drift"]["psi"]["critical"],
            line_dash="dot",
            line_color="#c96b72",
        )
    details = [
        heading(
            "Drift analysis",
            "Understand what changed, how much it changed, and whether performance followed.",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Feature", className="filter-label"),
                        dcc.Dropdown(
                            id={"type": "view-filter", "name": "feature"},
                            value=feature,
                            clearable=False,
                            options=[{"label": "All features", "value": "all"}]
                            + [
                                {"label": name, "value": name}
                                for name in [f"V{i}" for i in range(1, 29)] + ["Amount"]
                            ],
                            className="dropdown",
                        ),
                    ]
                )
            ],
            className="filters",
        ),
        html.Div(
            [
                metric(
                    "Window samples",
                    str(drift.get("sample_count", "—")),
                    stamp(drift.get("window_end")),
                    "#",
                ),
                metric(
                    "Labeled samples",
                    str(drift.get("labeled_count", "—")),
                    drift.get("concept_note", "No check yet"),
                    "✓",
                ),
                metric(
                    "Worst feature PSI",
                    number(data.get("overall_psi")),
                    data.get("severity", "Not available"),
                    "≋",
                ),
                metric(
                    "Hellinger distance",
                    number((drift.get("prediction_drift") or {}).get("hellinger_distance")),
                    "Prediction probability distribution",
                    "◎",
                ),
            ],
            className="metrics",
        ),
        card(
            "Which features shifted?",
            chart(fig, "feature-psi"),
            "Distribution shift by input feature",
        ),
        html.Div(
            card(
                "Feature statistics",
                table(
                    ["Feature", "PSI", "KL divergence"],
                    [
                        [name, number(value), number(data.get("feature_kl", {}).get(name))]
                        for name, value in sorted(
                            psi.items(), key=lambda item: item[1], reverse=True
                        )
                        if feature == "all" or name == feature
                    ],
                ),
                "Both metrics compare the same raw feature representation",
            ),
            style={"marginTop": "22px"},
        ),
        html.Div(
            [
                card("PSI history", trend(snapshot, "data")),
                card("Fraud F1 history", trend(snapshot, "concept")),
            ],
            className="grid-equal",
            style={"marginTop": "22px"},
        ),
        html.Div(
            drift.get("concept_note", "Generate traffic and run a drift check."), className="notice"
        ),
    ]
    return details


def predictions_view(snapshot):
    stats = snapshot["stats"]
    rows = snapshot["predictions"]
    fig = go.Figure()
    if rows:
        fig.add_trace(
            go.Histogram(
                x=[r["confidence"] for r in rows],
                nbinsx=25,
                marker_color=TEAL,
                name="Fraud probability",
            )
        )
    return [
        heading(
            "Predictions", "Inspect real inference events, source labels, probability, and latency."
        ),
        html.Div(
            [
                metric(
                    "Inference p50",
                    number(stats["p50_ms"], 2) + " ms",
                    "Recent retained records",
                    "↗",
                ),
                metric(
                    "Inference p95",
                    number(stats["p95_ms"], 2) + " ms",
                    "Model compute, per record",
                    "↗",
                ),
                metric(
                    "Inference p99",
                    number(stats["p99_ms"], 2) + " ms",
                    "Model compute, per record",
                    "↗",
                ),
                metric(
                    "Label coverage",
                    f"{stats['label_coverage']:.0%}",
                    f"{stats['labeled']} labeled records",
                    "✓",
                ),
            ],
            className="metrics",
        ),
        card("Fraud probability distribution", chart(fig, "confidence-hist")),
        html.Div(
            [html.H2("Prediction explorer"), command("Export CSV ↓", "export")],
            className="section-heading",
            style={"marginTop": "24px"},
        ),
        html.Div(
            table(
                [
                    "ID",
                    "Time · UTC",
                    "Version",
                    "Prediction",
                    "Fraud probability",
                    "True label",
                    "Source / scenario",
                    "Inference ms",
                ],
                [
                    [
                        r["id"],
                        stamp(r["timestamp"]),
                        "v" + r["model_version"],
                        "Fraud" if r["predicted_label"] else "Legitimate",
                        number(r["confidence"]),
                        "Pending" if r["true_label"] is None else str(r["true_label"]),
                        r["source"] + " / " + r["scenario"],
                        number(r["latency_ms"], 2),
                    ]
                    for r in rows
                ],
            ),
            className="card",
        ),
        html.P(
            "Explorer and CSV include up to 1,000 matching recent records. Latency percentiles use up to 5,000 records.",
            className="subtitle",
        ),
    ]


def models_view(snapshot):
    rows = snapshot["models"]
    running = busy(snapshot)
    blocks = [
        heading(
            "Models & retraining",
            "Every candidate earns promotion. Every deployment has a traceable version.",
        ),
        html.Div(
            [
                html.P(
                    "Automatic quality gates protect the champion. Rejection is a valid outcome.",
                    className="muted",
                ),
                command("Train challenger", "retrain", True, running),
            ],
            className="section-heading",
        ),
        card(
            "Model registry",
            table(
                ["Version", "Status", "Fraud F1", "Precision", "Recall", "Trained · UTC", "Action"],
                [
                    [
                        "v" + str(r["version"]),
                        badge(
                            "Champion" if r["is_production"] else "Registered",
                            "teal" if r["is_production"] else "blue",
                        ),
                        number(r["f1_score"]),
                        number(r["precision"]),
                        number(r["recall"]),
                        stamp(r["training_date"]),
                        "Active"
                        if r["is_production"]
                        else command("Rollback", "rollback-" + str(r["version"]), disabled=running),
                    ]
                    for r in rows
                ],
            ),
        ),
        html.Div(job_panel(snapshot), style={"marginTop": "22px"}),
        html.H2("Evaluation details", style={"marginTop": "24px", "marginBottom": "12px"}),
    ]
    evaluations = [j for j in snapshot["jobs"] if j["kind"] != "traffic"]
    if not evaluations:
        blocks.append(empty("No evaluation runs yet"))
    for job in evaluations[:10]:
        result = job.get("result", {})
        comparisons = []
        for dataset in ["historical", "recent"]:
            pair = result.get(dataset)
            if pair:
                comparisons.append(
                    table(
                        [dataset.title() + " holdout", "F1", "Recall", "Precision"],
                        [
                            [
                                "Champion" if len(pair) == 2 and i == 0 else "Challenger",
                                number(m["f1"]),
                                number(m["recall"]),
                                number(m["precision"]),
                            ]
                            for i, m in enumerate(pair)
                        ],
                    )
                )
        checks = result.get("checks", {})
        comparisons.append(
            html.Div(
                [
                    badge(
                        name.replace("_", " ") + (" · passed" if passed else " · failed"),
                        "teal" if passed else "red",
                    )
                    for name, passed in checks.items()
                ],
                className="mini-stats",
                style={"padding": "16px 20px"},
            )
        )
        comparisons.append(
            html.P(
                result.get("recent_evaluation_note", result.get("message", "Waiting for worker")),
                className="subtitle",
                style={"padding": "0 20px 15px"},
            )
        )
        blocks.append(
            html.Details(
                [
                    html.Summary(job["status"] + " · " + stamp(job["created_at"])),
                    *comparisons,
                    html.Details(
                        [
                            html.Summary("Technical run details"),
                            html.Pre(json.dumps(result, indent=2)),
                        ]
                    ),
                ],
                className="card job-detail",
                open=job == evaluations[0],
            )
        )
    return blocks


def alerts_view(snapshot, search):
    needle = (search or "").lower()
    rows = [
        a
        for a in snapshot["alerts"]
        if needle in (a["message"] + a["severity"] + a["drift_type"]).lower()
    ]
    return [
        heading("Alert center", "Understand the signal. Acknowledge what you have reviewed."),
        dcc.Input(
            id={"type": "view-filter", "name": "search"},
            type="search",
            value=search,
            placeholder="Search alerts…",
            debounce=True,
            className="search",
        ),
        html.Div(
            table(
                ["Severity", "Message", "Delivery", "Occurrences", "Time · UTC", "Action"],
                [
                    [
                        badge(
                            r["severity"],
                            "red"
                            if r["severity"] == "CRITICAL"
                            else "orange"
                            if r["severity"] == "WARNING"
                            else "teal",
                        ),
                        r["message"],
                        ", ".join(
                            k + ": " + v for k, v in json.loads(r["delivery_json"] or "{}").items()
                        ),
                        r["occurrences"],
                        stamp(r["timestamp"]),
                        "Acknowledged"
                        if r["acknowledged"]
                        else command("Acknowledge", "ack-" + str(r["id"])),
                    ]
                    for r in rows
                ],
            ),
            className="card",
            style={"marginTop": "20px"},
        ),
    ]


def settings_view(snapshot):
    definitions = lambda values: html.Div(
        [
            html.Div(
                [html.Span(k.replace("_", " ").title()), html.Strong(str(v))],
                className="definition",
            )
            for k, v in values.items()
        ],
        className="card-body",
    )
    return [
        heading("Workspace settings", "Configuration visibility without exposing credentials."),
        html.Div(
            [
                card("Promotion gates", definitions(snapshot["promotion"])),
                card(
                    "Notification channels",
                    definitions(
                        {
                            k: "Configured" if v else "Disabled"
                            for k, v in snapshot["notifications"].items()
                        }
                    ),
                ),
                card(
                    "Feature PSI thresholds",
                    definitions(snapshot["thresholds"]["data_drift"]["psi"]),
                ),
                card(
                    "Workspace",
                    definitions(
                        {
                            "dataset": snapshot["source"],
                            "serving_workers": 1,
                            "telemetry_retention": "30 days / 50,000 predictions",
                            "time_zone": "UTC",
                        }
                    ),
                ),
            ],
            className="settings-grid",
        ),
        html.P(
            "Edit configs/base_config.yaml, configs/drift_thresholds.yaml, or environment variables and restart services to change configuration. SMTP and Slack secrets stay server-side.",
            className="notice",
            style={"marginTop": "22px"},
        ),
        card(
            "Reset generated demo traffic",
            html.Div(
                [
                    html.P(
                        "Clears predictions labeled demo. Models, user predictions, source datasets, and audit history are preserved.",
                        className="subtitle",
                    ),
                    command(
                        "Reset demo telemetry", "reset", disabled=busy(snapshot), extra="btn-danger"
                    ),
                ],
                className="card-body",
            ),
        ),
    ]
