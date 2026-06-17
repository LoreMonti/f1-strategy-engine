# =========================================================
# Interactive Race Strategy Dashboard — Plotly / Dash
# =========================================================

from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.strategy import RaceResult
from src.models.multi_car import MultiCarRaceResult

# ── Dashboard palette ─────────────────────────────────────────────────────────
_DARK_BG = "#111"

# ── F1 compound colours ───────────────────────────────────────────────────────
_COMPOUND_COLOUR = {
    "Soft":   "#e8002d",
    "Medium": "#ffd700",
    "Hard":   "#b2b2b2",
    "Inter":  "#39b54a",
    "Wet":    "#0067ff",
}
_FALLBACK_COLOURS = [
    "#636efa", "#ef553b", "#00cc96", "#ab63fa",
    "#ffa15a", "#19d3f3", "#ff6692", "#b6e880",
]

_STRATEGY_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _compound_colour(name: str, fallback_idx: int = 0) -> str:
    return _COMPOUND_COLOUR.get(name, _FALLBACK_COLOURS[fallback_idx % len(_FALLBACK_COLOURS)])


def _strategy_colour(idx: int) -> str:
    return _STRATEGY_COLOURS[idx % len(_STRATEGY_COLOURS)]


# ── Virtual cumulative helper ─────────────────────────────────────────────────

def _virtual_cum(result: RaceResult, lap_index: int) -> float:
    return result.laps[lap_index].cumulative_time - sum(
        lr.pit_time_loss for lr in result.laps[: lap_index + 1]
    )


# ── Figure builders ───────────────────────────────────────────────────────────

def _fig_gap(results: list[RaceResult]) -> go.Figure:
    ref = results[0]
    num_laps = ref.num_laps
    laps = list(range(1, num_laps + 1))

    fig = go.Figure()
    for i, r in enumerate(results):
        gaps = [_virtual_cum(r, j) - _virtual_cum(ref, j) for j in range(num_laps)]
        pit_laps = {lr.lap for lr in r.laps if lr.pit_stop}
        fig.add_trace(go.Scatter(
            x=laps, y=gaps,
            mode="lines+markers",
            name=r.strategy,
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(
                size=[10 if lap in pit_laps else 5 for lap in laps],
                symbol=["diamond" if lap in pit_laps else "circle" for lap in laps],
            ),
            hovertemplate="Lap %{x}<br>Gap: %{y:.3f} s<extra>%{fullData.name}</extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Virtual Strategy Gap (diamonds = pit stop laps)",
        xaxis_title="Lap",
        yaxis_title="Gap to Leader [s]",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def _fig_lap_times(results: list[RaceResult]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        laps = [lr.lap for lr in r.laps]
        times = [lr.raw_lap_time for lr in r.laps]
        compounds = [lr.compound for lr in r.laps]
        pit_laps = {lr.lap for lr in r.laps if lr.pit_stop}

        fig.add_trace(go.Scatter(
            x=laps, y=times,
            mode="lines+markers",
            name=r.strategy,
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(
                size=[10 if lap in pit_laps else 5 for lap in laps],
                symbol=["diamond" if lap in pit_laps else "circle" for lap in laps],
            ),
            customdata=compounds,
            hovertemplate=(
                "Lap %{x}<br>Time: %{y:.3f} s<br>Compound: %{customdata}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        title="Raw Lap Time Evolution",
        xaxis_title="Lap",
        yaxis_title="Raw Lap Time [s]",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def _fig_tyre_wear(results: list[RaceResult]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        laps = [lr.lap for lr in r.laps]
        wear = [min(lr.final_tyre_wear, 1.0) * 100 for lr in r.laps]
        compounds = [lr.compound for lr in r.laps]

        fig.add_trace(go.Scatter(
            x=laps, y=wear,
            mode="lines+markers",
            name=r.strategy,
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(size=4),
            customdata=compounds,
            hovertemplate=(
                "Lap %{x}<br>Wear: %{y:.1f}%<br>Compound: %{customdata}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.add_hline(y=80, line_dash="dot", line_color="orange", opacity=0.6,
                  annotation_text="80% cliff", annotation_position="top right")
    fig.update_layout(
        title="Tyre Wear Evolution",
        xaxis_title="Lap",
        yaxis_title="Tyre Wear [%]",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def _fig_tyre_temp(results: list[RaceResult]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        laps = [lr.lap for lr in r.laps]
        temp = [lr.final_tyre_temperature for lr in r.laps]
        compounds = [lr.compound for lr in r.laps]

        fig.add_trace(go.Scatter(
            x=laps, y=temp,
            mode="lines+markers",
            name=r.strategy,
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(size=4),
            customdata=compounds,
            hovertemplate=(
                "Lap %{x}<br>Temp: %{y:.1f} °C<br>Compound: %{customdata}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        title="Tyre Temperature Evolution",
        xaxis_title="Lap",
        yaxis_title="Tyre Temperature [°C]",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def _fig_telemetry(telemetry: dict) -> go.Figure:
    s = telemetry["s"]

    panels = [
        ("Speed [km/h]",          [("v_kmh",                    "royalblue",   None)]),
        ("Throttle / Brake [%]",  [("throttle",                 "limegreen",   "Throttle"),
                                   ("brake",                    "crimson",     "Brake")]),
        ("Gear",                  [("gear",                     "darkorange",  None)]),
        ("RPM",                   [("rpm",                      "plum",        None)]),
        ("Tyre Wear [%]",         [("front_tyre_wear",          "steelblue",   "Front"),
                                   ("rear_tyre_wear",           "tomato",      "Rear")]),
        ("Tyre Temperature [°C]", [("front_tyre_temperature",   "steelblue",   "Front"),
                                   ("rear_tyre_temperature",    "tomato",      "Rear")]),
        ("Grip Multiplier",       [("front_grip_multiplier",    "steelblue",   "Front"),
                                   ("rear_grip_multiplier",     "tomato",      "Rear")]),
    ]

    fig = make_subplots(
        rows=len(panels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[p[0] for p in panels],
    )

    for row, (ylabel, channels) in enumerate(panels, start=1):
        for key, colour, label in channels:
            raw = telemetry.get(key, [])
            if key in ("front_tyre_wear", "rear_tyre_wear"):
                import numpy as np
                raw = list(np.array(raw) * 100.0)
            fig.add_trace(
                go.Scatter(
                    x=s, y=raw,
                    mode="lines",
                    line=dict(color=colour, width=1.5),
                    name=label or ylabel,
                    showlegend=(label is not None),
                    hovertemplate=f"{ylabel}: %{{y:.2f}}<extra></extra>",
                ),
                row=row, col=1,
            )

    fig.update_xaxes(title_text="Distance [m]", row=len(panels), col=1)
    fig.update_layout(
        height=1100,
        title="Lap Telemetry",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.04),
    )
    return fig


def _fig_stint_bars(results: list[RaceResult]) -> go.Figure:
    """Horizontal Gantt-style stint bars for the top strategies."""
    fig = go.Figure()
    strategy_labels = [r.strategy for r in results]
    seen_compounds: set[str] = set()

    for i, r in enumerate(results):
        for stint in r.stints:
            cname = stint.compound.name
            colour = _compound_colour(cname, list(seen_compounds).index(cname) if cname in seen_compounds else len(seen_compounds))
            seen_compounds.add(cname)
            show_legend = cname not in {t.compound.name for rr in results[:i] for t in rr.stints}

            fig.add_trace(go.Bar(
                x=[stint.length],
                y=[r.strategy],
                base=[stint.start_lap - 1],
                orientation="h",
                name=cname,
                marker_color=colour,
                showlegend=show_legend,
                hovertemplate=(
                    f"{cname}: L{stint.start_lap}–L{stint.end_lap} "
                    f"({stint.length} laps)<extra>{r.strategy}</extra>"
                ),
                legendgroup=cname,
            ))

    fig.update_layout(
        barmode="stack",
        title="Stint Breakdown",
        xaxis_title="Lap",
        yaxis_title="Strategy",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
        height=max(300, 80 + len(results) * 45),
    )
    return fig


# ── Dashboard builder ─────────────────────────────────────────────────────────

def _fig_mc_positions(mc: MultiCarRaceResult) -> go.Figure:
    """Position chart: lap vs race position for each car."""
    fig = go.Figure()
    for i, car in enumerate(mc.cars):
        laps = [lr.lap for lr in car.laps]
        positions = [lr.position for lr in car.laps]
        pit_laps = {lr.lap for lr in car.laps if lr.pit_stop}
        fig.add_trace(go.Scatter(
            x=laps, y=positions,
            mode="lines+markers",
            name=f"P{car.grid_position} {car.name} ({car.strategy_name})",
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(
                size=[10 if lap in pit_laps else 5 for lap in laps],
                symbol=["diamond" if lap in pit_laps else "circle" for lap in laps],
            ),
            hovertemplate="Lap %{x}<br>Position: P%{y}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title="Race Position Evolution (diamonds = pit stop)",
        xaxis_title="Lap",
        yaxis_title="Position",
        yaxis=dict(autorange="reversed", tickmode="linear", dtick=1),
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified",
    )
    return fig


def _fig_mc_gaps(mc: MultiCarRaceResult) -> go.Figure:
    """Gap to leader evolution for each car."""
    fig = go.Figure()
    for i, car in enumerate(mc.cars):
        laps = [lr.lap for lr in car.laps]
        gaps = [lr.gap_to_leader for lr in car.laps]
        fig.add_trace(go.Scatter(
            x=laps, y=gaps,
            mode="lines+markers",
            name=f"P{car.grid_position} {car.name}",
            line=dict(color=_strategy_colour(i), width=2),
            marker=dict(size=4),
            hovertemplate="Lap %{x}<br>Gap: %{y:.3f} s<extra>%{fullData.name}</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Gap to Race Leader",
        xaxis_title="Lap",
        yaxis_title="Gap [s]",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified",
    )
    return fig


def _fig_mc_traffic(mc: MultiCarRaceResult) -> go.Figure:
    """Cumulative traffic penalty per car."""
    cars_sorted = sorted(mc.cars, key=lambda c: c.grid_position)
    names = [f"P{c.grid_position} {c.name}" for c in cars_sorted]
    penalties = [c.total_traffic_penalty for c in cars_sorted]
    colours = [_strategy_colour(i) for i in range(len(cars_sorted))]
    fig = go.Figure(go.Bar(
        x=names, y=penalties,
        marker_color=colours,
        hovertemplate="%{x}<br>Traffic penalty: %{y:.2f} s<extra></extra>",
    ))
    fig.update_layout(
        title="Total Traffic Penalty per Car",
        xaxis_title="Car",
        yaxis_title="Penalty [s]",
        template="plotly_dark",
    )
    return fig


def _fig_monte_carlo(distributions) -> go.Figure:
    """
    Monte Carlo robustness, three decision-oriented views:
      1. Win probability    — which strategy finishes ahead.
      2. Risk vs reward      — median race time vs worst-case (P95), bubble = win%.
      3. Outcome range       — P5–P95 distribution (the long tail is shared SC risk).
    """
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.26, 0.42, 0.32],
        vertical_spacing=0.11,
        subplot_titles=[
            "Win probability — which strategy finishes ahead",
            "Risk vs reward — lower-left is better (bubble = win %)",
            "Outcome range P5–P95 — the long upper tail is shared Safety-Car risk",
        ],
    )

    n = len(distributions)
    colours = [_strategy_colour(i) for i in range(n)]

    # ── 1. Win-probability horizontal bars (ranked) ───────────────────────────
    ranked = sorted(range(n), key=lambda i: distributions[i].win_probability)
    fig.add_trace(go.Bar(
        x=[distributions[i].win_probability * 100 for i in ranked],
        y=[distributions[i].name for i in ranked],
        orientation="h",
        marker_color=[colours[i] for i in ranked],
        text=[f"{distributions[i].win_probability * 100:.0f}%" for i in ranked],
        textposition="auto",
        showlegend=False,
        hovertemplate="%{y}<br>%{x:.1f}% of races won<extra></extra>",
    ), row=1, col=1)

    # ── 2. Risk-reward bubble scatter (one trace per strategy → legend) ───────
    for i, d in enumerate(distributions):
        short = d.name.replace("DP ", "")
        fig.add_trace(go.Scatter(
            x=[d.p50 / 60.0], y=[d.p95 / 60.0],
            mode="markers", name=short,
            marker=dict(size=10 + d.win_probability * 45, color=colours[i],
                        line=dict(color="#fff", width=1), opacity=0.85),
            legendgroup=d.name,
            hovertemplate=(d.name + "<br>median %{x:.2f} min · "
                           "P95 %{y:.2f} min · "
                           f"win {d.win_probability*100:.0f}%<extra></extra>"),
        ), row=2, col=1)

    # ── 3. Outcome range: box plot per strategy from the real samples ─────────
    for i, d in enumerate(distributions):
        fig.add_trace(go.Box(
            x=d.samples / 60.0, name=d.name, orientation="h",
            marker_color=colours[i], line_color=colours[i],
            boxpoints=False, showlegend=False, legendgroup=d.name,
            hovertemplate=(d.name + "<br>median "
                           f"{d.p50/60:.2f} min · P5 {d.p5/60:.2f} · "
                           f"P95 {d.p95/60:.2f}<extra></extra>"),
        ), row=3, col=1)

    sc_pct  = distributions[0].sc_exposure_pct if distributions else 0.0
    neu_pct = distributions[0].neutralisation_pct if distributions else 0.0
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
        height=860,
        title=dict(
            text=(f"<b>Monte Carlo — Strategy Robustness</b>  "
                  f"<sup>{neu_pct:.0f}% of races had a neutralisation "
                  f"({sc_pct:.0f}% a full SC) · "
                  f"win-probability via paired sampling (common random numbers)</sup>"),
            font=dict(size=14), y=0.985,
        ),
        # Legend at the very bottom so it never overlaps the subplot titles.
        legend=dict(orientation="h", yanchor="top", y=-0.07,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=90, b=70),
    )
    fig.update_xaxes(title_text="win probability [%]", row=1, col=1)
    fig.update_xaxes(title_text="median race time [min]  →  slower", row=2, col=1)
    fig.update_yaxes(title_text="worst case P95 [min]  →  riskier", row=2, col=1)
    fig.update_xaxes(title_text="race time [min]", row=3, col=1)
    return fig


def build_dashboard(
    race_results: list[RaceResult],
    telemetry: dict,
    multi_lap_res: dict,
    circuit_name: str = "F1",
    mc_result: MultiCarRaceResult | None = None,
    animation_fig=None,
    f1_comparison: dict | None = None,
    mc_distributions=None,
) -> dash.Dash:
    app = dash.Dash(__name__, title=f"F1 Sim — {circuit_name}")

    max_n = len(race_results)
    default_n = min(5, max_n)

    app.layout = html.Div(
        style={"backgroundColor": "#111", "color": "#eee", "fontFamily": "monospace", "padding": "16px"},
        children=[
            html.H2(
                f"F1 Race Strategy Dashboard — {circuit_name}",
                style={"textAlign": "center", "marginBottom": "8px"},
            ),

            dcc.Tabs(
                id="main-tabs",
                value="strategy",
                colors={"border": "#444", "primary": "#e8002d", "background": "#1a1a1a"},
                children=[
                    # ── Strategy tab ──────────────────────────────────────
                    dcc.Tab(
                        label="Race Strategy",
                        value="strategy",
                        style={"color": "#ccc", "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center",
                                       "gap": "12px", "margin": "12px 0"},
                                children=[
                                    html.Label("Top N strategies:", style={"whiteSpace": "nowrap"}),
                                    dcc.Slider(
                                        id="top-n-slider",
                                        min=2, max=max_n, step=1, value=default_n,
                                        marks={i: str(i) for i in range(2, max_n + 1)},
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                            ),
                            # 2×2 grid: gap, lap times, wear, temp
                            html.Div(
                                style={"display": "grid",
                                       "gridTemplateColumns": "1fr 1fr",
                                       "gap": "8px"},
                                children=[
                                    dcc.Graph(id="graph-gap"),
                                    dcc.Graph(id="graph-lap-times"),
                                    dcc.Graph(id="graph-wear"),
                                    dcc.Graph(id="graph-temp"),
                                ],
                            ),
                            dcc.Graph(id="graph-stints"),
                        ],
                    ),

                    # ── Telemetry tab ─────────────────────────────────────
                    dcc.Tab(
                        label="Lap Telemetry",
                        value="telemetry",
                        style={"color": "#ccc", "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=[
                            dcc.Graph(
                                id="graph-telemetry",
                                figure=_fig_telemetry(telemetry),
                                style={"height": "1100px"},
                            ),
                        ],
                    ),

                    # ── Lap Animation tab ─────────────────────────────────
                    dcc.Tab(
                        label="Lap Animation" + ("" if animation_fig else " (run with --dashboard)"),
                        value="animation",
                        disabled=(animation_fig is None),
                        style={"color": "#888" if animation_fig is None else "#ccc",
                               "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=([] if animation_fig is None else [
                            dcc.Graph(
                                figure=animation_fig,
                                style={"height": "720px"},
                                config={"displayModeBar": True},
                            ),
                        ]),
                    ),

                    # ── FastF1 comparison tab ─────────────────────────────
                    dcc.Tab(
                        label="FastF1 vs Sim" + ("" if f1_comparison else " (needs --dashboard)"),
                        value="f1compare",
                        disabled=(f1_comparison is None),
                        style={"color": "#888" if f1_comparison is None else "#ccc",
                               "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=([] if f1_comparison is None else [
                            dcc.Graph(figure=f1_comparison["tel_fig"],
                                      style={"height": "620px"}),
                            dcc.Graph(figure=f1_comparison["race_fig"],
                                      style={"height": "570px"}),
                        ]),
                    ),

                    # ── Multi-car tab ──────────────────────────────────────
                    dcc.Tab(
                        label="Multi-Car Race" + ("" if mc_result else " (run with --multi-car)"),
                        value="multicar",
                        disabled=(mc_result is None),
                        style={"color": "#888" if mc_result is None else "#ccc",
                               "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=([] if mc_result is None else [
                            html.Div(
                                style={"display": "grid",
                                       "gridTemplateColumns": "1fr 1fr",
                                       "gap": "8px",
                                       "marginTop": "12px"},
                                children=[
                                    dcc.Graph(figure=_fig_mc_positions(mc_result)),
                                    dcc.Graph(figure=_fig_mc_gaps(mc_result)),
                                ],
                            ),
                            dcc.Graph(figure=_fig_mc_traffic(mc_result)),
                        ]),
                    ),

                    # ── Monte Carlo robustness tab ────────────────────────
                    dcc.Tab(
                        label="Monte Carlo (Risk)" + ("" if mc_distributions else " (n/a)"),
                        value="montecarlo",
                        disabled=(mc_distributions is None),
                        style={"color": "#888" if mc_distributions is None else "#ccc",
                               "backgroundColor": "#1a1a1a"},
                        selected_style={"color": "#fff", "backgroundColor": "#333"},
                        children=([] if mc_distributions is None else [
                            dcc.Graph(figure=_fig_monte_carlo(mc_distributions),
                                      style={"height": "780px"}),
                        ]),
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("graph-gap",       "figure"),
        Output("graph-lap-times", "figure"),
        Output("graph-wear",      "figure"),
        Output("graph-temp",      "figure"),
        Output("graph-stints",    "figure"),
        Input("top-n-slider",     "value"),
    )
    def update_strategy_tab(top_n: int):
        selected = race_results[:top_n]
        return (
            _fig_gap(selected),
            _fig_lap_times(selected),
            _fig_tyre_wear(selected),
            _fig_tyre_temp(selected),
            _fig_stint_bars(selected),
        )

    return app


def launch(
    race_results: list[RaceResult],
    telemetry: dict,
    multi_lap_res: dict,
    circuit_name: str = "F1",
    port: int = 8050,
    debug: bool = False,
    mc_result: MultiCarRaceResult | None = None,
    animation_fig=None,
    f1_comparison: dict | None = None,
    mc_distributions=None,
) -> None:
    app = build_dashboard(race_results, telemetry, multi_lap_res, circuit_name,
                          mc_result, animation_fig, f1_comparison,
                          mc_distributions=mc_distributions)
    print(f"\nDashboard running at http://127.0.0.1:{port}/  (Ctrl+C to stop)\n")
    app.run(debug=debug, port=port)
