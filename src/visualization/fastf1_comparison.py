# =========================================================
# FastF1 vs Simulator Comparison Figures
#
# Three panels:
#   1. Speed trace overlay    (real Q lap vs simulated)
#   2. Throttle/Brake overlay (real Q lap vs simulated)
#   3. Race lap time evolution (real winner vs simulated best strategy)
# =========================================================

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.strategy import RaceResult

_REAL_COL = "#e8002d"    # F1 red  — real data
_SIM_COL  = "#4fc3f7"    # cyan    — simulated data
_DARK_BG  = "#111111"


# ── Telemetry comparison ──────────────────────────────────────────────────────

def fig_telemetry_comparison(
    sim_telemetry: dict,
    real_telemetry: dict,
    circuit_name: str = "",
) -> go.Figure:
    """
    Overlay real qualifying lap vs simulated lap.

    Panels (shared x = distance [m]):
      Row 1: Speed [km/h]
      Row 2: Throttle [%]
      Row 3: Brake [%]
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Speed [km/h]", "Throttle [%]", "Brake [%]"],
    )

    s_sim  = np.array(sim_telemetry["s"])
    s_real = np.array(real_telemetry["s"])

    # ── Speed ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s_real, y=real_telemetry["speed_kmh"],
        name=f"Real — {real_telemetry['driver']} ({real_telemetry['lap_time_s']:.3f}s)",
        line=dict(color=_REAL_COL, width=2),
        hovertemplate="dist %{x:.0f}m<br>%{y:.1f} km/h<extra>Real</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=s_sim, y=np.array(sim_telemetry["v_kmh"]),
        name=f"Sim — {sim_telemetry.get('lap_time_s', 0):.3f}s",
        line=dict(color=_SIM_COL, width=2),
        hovertemplate="dist %{x:.0f}m<br>%{y:.1f} km/h<extra>Sim</extra>",
    ), row=1, col=1)

    # ── Throttle ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s_real, y=real_telemetry["throttle"] * 100,
        name="Throttle Real", showlegend=False,
        line=dict(color=_REAL_COL, width=1.5),
        hovertemplate="dist %{x:.0f}m<br>%{y:.0f}%<extra>Real Throttle</extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=s_sim, y=np.array(sim_telemetry["throttle"]),  # already 0-100
        name="Throttle Sim", showlegend=False,
        line=dict(color=_SIM_COL, width=1.5),
        hovertemplate="dist %{x:.0f}m<br>%{y:.0f}%<extra>Sim Throttle</extra>",
    ), row=2, col=1)

    # ── Brake ─────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s_real, y=real_telemetry["brake"] * 100,
        name="Brake Real", showlegend=False,
        line=dict(color=_REAL_COL, width=1.5),
        hovertemplate="dist %{x:.0f}m<br>%{y:.0f}%<extra>Real Brake</extra>",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=s_sim, y=np.array(sim_telemetry["brake"]),  # already 0-100
        name="Brake Sim", showlegend=False,
        line=dict(color=_SIM_COL, width=1.5),
        hovertemplate="dist %{x:.0f}m<br>%{y:.0f}%<extra>Sim Brake</extra>",
    ), row=3, col=1)

    # ── Speed delta annotation ────────────────────────────────────────────────
    real_max = float(np.nanmax(real_telemetry["speed_kmh"]))
    sim_max  = float(np.nanmax(sim_telemetry["v_kmh"]))
    real_avg = float(np.nanmean(real_telemetry["speed_kmh"]))
    sim_avg  = float(np.nanmean(sim_telemetry["v_kmh"]))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        height=600,
        title=dict(
            text=f"<b>Qualifying Telemetry — Sim vs Real</b>  {circuit_name}<br>"
                 f"<sup>Real Vmax {real_max:.0f} km/h  Sim Vmax {sim_max:.0f} km/h  |  "
                 f"Real Vavg {real_avg:.0f}  Sim Vavg {sim_avg:.0f}</sup>",
            font=dict(size=14),
        ),
        legend=dict(orientation="h", y=1.06, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Distance [m]", row=3, col=1)

    return fig


# ── Race lap time comparison ──────────────────────────────────────────────────

_COMPOUND_COLOUR = {
    "Soft": "#e8002d", "Medium": "#ffd700",
    "Hard": "#b2b2b2", "Unknown": "#888888",
}


def fig_race_laps_comparison(
    sim_result: RaceResult,
    real_laps: dict,
    circuit_name: str = "",
    sim_wet: bool = False,
) -> go.Figure:
    """
    Compare simulated best strategy lap times vs real race winner.

    Shows:
      - Scatter: lap time per lap (coloured by compound)
      - Delta subplot: sim − real per lap
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Lap Time [s]", "Δ Sim − Real [s]"],
    )

    # ── Real laps ─────────────────────────────────────────────────────────────
    real_by_lap = {l["lap"]: l for l in real_laps["laps"]}
    real_laps_sorted = sorted(real_laps["laps"], key=lambda x: x["lap"])

    seen_compounds: set[str] = set()
    for lap_info in real_laps_sorted:
        cmp  = lap_info["compound"]
        col  = _COMPOUND_COLOUR.get(cmp, "#888")
        show = cmp not in seen_compounds
        seen_compounds.add(cmp)
        fig.add_trace(go.Scatter(
            x=[lap_info["lap"]], y=[lap_info["lap_time_s"]],
            mode="markers",
            marker=dict(color=col, size=8, symbol="circle"),
            name=f"Real — {cmp}",
            legendgroup=f"real_{cmp}",
            showlegend=show,
            hovertemplate=(
                f"Lap %{{x}}  {cmp}<br>"
                f"Time: %{{y:.3f}}s<extra>Real {real_laps['driver']}</extra>"
            ),
        ), row=1, col=1)

    # Connect real dots with a grey line
    real_laps_arr = [(l["lap"], l["lap_time_s"]) for l in real_laps_sorted]
    if real_laps_arr:
        rx, ry = zip(*real_laps_arr)
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines",
            line=dict(color="rgba(200,200,200,0.4)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # ── Simulated laps ────────────────────────────────────────────────────────
    seen_sim: set[str] = set()
    for lr in sim_result.laps:
        cmp  = lr.compound
        col  = _COMPOUND_COLOUR.get(cmp, "#4fc3f7")
        show = f"sim_{cmp}" not in seen_sim
        seen_sim.add(f"sim_{cmp}")
        sym  = "diamond" if lr.pit_stop else "circle"
        fig.add_trace(go.Scatter(
            x=[lr.lap], y=[lr.raw_lap_time],
            mode="markers",
            marker=dict(color=col, size=9, symbol=sym,
                        line=dict(color=_SIM_COL, width=1.5)),
            name=f"Sim — {cmp}",
            legendgroup=f"sim_{cmp}",
            showlegend=show,
            hovertemplate=(
                f"Lap %{{x}}  {cmp}<br>"
                f"Time: %{{y:.3f}}s<extra>Sim</extra>"
            ),
        ), row=1, col=1)

    # Simulated line
    sim_x = [lr.lap for lr in sim_result.laps]
    sim_y = [lr.raw_lap_time for lr in sim_result.laps]
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_y, mode="lines",
        line=dict(color=f"rgba(79,195,247,0.5)", width=1),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── Delta plot ────────────────────────────────────────────────────────────
    # Exclude pit-stop laps and SC/VSC outliers so they don't distort the delta.
    _median_real = real_laps.get("median_lap_s", real_laps["avg_lap_s"])
    _delta_threshold = _median_real * 1.10

    deltas_x, deltas_y = [], []
    for lr in sim_result.laps:
        if lr.lap in real_by_lap:
            rl = real_by_lap[lr.lap]
            if rl.get("pit_in", False) or rl["lap_time_s"] > _delta_threshold:
                continue
            delta = lr.raw_lap_time - rl["lap_time_s"]
            deltas_x.append(lr.lap)
            deltas_y.append(delta)

    if deltas_y:
        colours = [_REAL_COL if d > 0 else _SIM_COL for d in deltas_y]
        fig.add_trace(go.Bar(
            x=deltas_x, y=deltas_y,
            marker_color=colours,
            name="Δ Sim−Real",
            showlegend=False,
            hovertemplate="Lap %{x}<br>Δ = %{y:+.3f}s<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="white",
                      opacity=0.4, row=2, col=1)

        avg_delta = float(np.mean(deltas_y))
        fig.add_annotation(
            text=f"Avg Δ = {avg_delta:+.2f}s/lap",
            x=0.98, y=0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(color="white", size=12),
            xanchor="right",
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    sim_avg  = float(np.mean([lr.raw_lap_time for lr in sim_result.laps]))
    real_avg = real_laps["avg_lap_s"]
    sim_fast = sim_result.fastest_lap.raw_lap_time
    real_fast = real_laps["fastest_lap_s"]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        height=550,
        title=dict(
            text=(
                f"<b>Race Lap Times — Sim vs Real ({real_laps['driver']})</b>  {circuit_name}"
                + (
                    ("  🌧️ WET/MIXED RACE — sim weather model active"
                     if sim_wet else
                     "  ⚠️ WET/MIXED RACE — comparison not valid (sim is dry)")
                    if real_laps.get("wet_race") else ""
                )
                + f"<br><sup>"
                f"Sim fastest {sim_fast:.3f}s  avg {sim_avg:.3f}s  |  "
                f"Real fastest {real_fast:.3f}s  avg {real_avg:.3f}s  |  "
                f"Avg gap {sim_avg - real_avg:+.2f}s/lap"
                f"</sup>"
            ),
            font=dict(size=13),
        ),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
        bargap=0.1,
    )
    fig.update_xaxes(title_text="Lap", row=2, col=1)
    fig.update_yaxes(title_text="Lap Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Δ [s]", row=2, col=1)

    return fig
