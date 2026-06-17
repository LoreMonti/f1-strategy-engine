# =========================================================
# Track Animator
#
# Reconstructs approximate 2D track geometry from mini-sector
# data (lateral_g + turn direction) and builds an animated
# Plotly figure showing:
#   - Car moving along the track map
#   - Speed trace with cursor
#   - Live throttle / brake / gear display
# =========================================================

from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go


_G = 9.81   # m/s²


# ── Track geometry reconstruction ────────────────────────────────────────────

def _reconstruct_track(
    mini_sectors: list[dict],
    total_length_m: float,
    samples_per_segment: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct 2D track coordinates from mini-sector heading_change_deg data.

    Each segment is a circular arc (or straight) defined by:
      - length   : distance [m] between consecutive dist_m values
      - heading_change_deg : total signed heading change through this segment
                             (negative = right/clockwise, positive = left)

    The approach is purely geometric: no physics needed. The total heading
    change over the full lap must equal -360° for a clockwise circuit.

    Parameters
    ----------
    mini_sectors : raw YAML rows, each must have dist_m and heading_change_deg
    total_length_m : total lap distance [m]
    samples_per_segment : points per segment (for smooth arc rendering)

    Returns
    -------
    s_arr, x_arr, y_arr : parallel arrays of (distance, x, y)
    """
    dists   = [ms["dist_m"] for ms in mini_sectors]
    lengths = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
    lengths.append(total_length_m - dists[-1])

    s_list, x_list, y_list = [], [], []

    cx, cy  = 0.0, 0.0
    heading = 0.0          # radians; 0 = east, increases counterclockwise

    for ms, seg_len in zip(mini_sectors, lengths):
        hc_deg = float(ms.get("heading_change_deg", 0.0))
        hc_rad = math.radians(hc_deg)
        s0     = float(ms["dist_m"])
        n      = max(2, samples_per_segment)

        if abs(hc_rad) < 1e-3:
            # ── Straight segment ──────────────────────────────────────────
            for j in range(n):
                t = j / n
                s_list.append(s0 + t * seg_len)
                x_list.append(cx + t * seg_len * math.cos(heading))
                y_list.append(cy + t * seg_len * math.sin(heading))
            cx += seg_len * math.cos(heading)
            cy += seg_len * math.sin(heading)

        else:
            # ── Circular arc ──────────────────────────────────────────────
            # Radius derived from arc length and total heading change
            r    = seg_len / abs(hc_rad)
            sign = 1 if hc_deg > 0 else -1      # +1 left, -1 right

            # Centre of curvature: perpendicular to current heading
            c_angle = heading + sign * math.pi / 2
            rcx = cx + r * math.cos(c_angle)
            rcy = cy + r * math.sin(c_angle)

            start_arm = math.atan2(cy - rcy, cx - rcx)

            for j in range(n):
                t     = j / n
                angle = start_arm + sign * t * abs(hc_rad)
                s_list.append(s0 + t * seg_len)
                x_list.append(rcx + r * math.cos(angle))
                y_list.append(rcy + r * math.sin(angle))

            end_arm  = start_arm + sign * abs(hc_rad)
            cx       = rcx + r * math.cos(end_arm)
            cy       = rcy + r * math.sin(end_arm)
            heading += hc_rad

    # Close the loop
    s_list.append(float(total_length_m))
    x_list.append(x_list[0])
    y_list.append(y_list[0])

    s_arr = np.array(s_list)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    # ── Linear closure correction ────────────────────────────────────────────
    # Apply correction BEFORE closing, so the last reconstructed point
    # is nudged back to (x[0], y[0]) via a linear drift along the lap.
    dx_err = x_arr[-1] - x_arr[0]
    dy_err = y_arr[-1] - y_arr[0]
    fractions = s_arr / total_length_m
    x_arr -= fractions * dx_err
    y_arr -= fractions * dy_err

    # Close the loop by appending first point at the end
    s_arr = np.append(s_arr, total_length_m)
    x_arr = np.append(x_arr, x_arr[0])
    y_arr = np.append(y_arr, y_arr[0])

    # ── Rotate so pit straight is horizontal (heading → 0) ───────────────────
    # The first two points define the pit straight direction.
    if len(x_arr) > 1:
        dx0 = x_arr[min(20, len(x_arr) - 1)] - x_arr[0]
        dy0 = y_arr[min(20, len(y_arr) - 1)] - y_arr[0]
        rot = -math.atan2(dy0, dx0)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        x_rot = x_arr * cos_r - y_arr * sin_r
        y_rot = x_arr * sin_r + y_arr * cos_r
        x_arr, y_arr = x_rot, y_rot

    return s_arr, x_arr, y_arr


# ── Animation builder ─────────────────────────────────────────────────────────

def build_lap_animation(
    telemetry: dict,
    mini_sectors: list[dict],
    total_length_m: float,
    circuit_name: str = "F1 Circuit",
    frame_step: int = 10,       # subsample telemetry: every N points ≈ N×5 m
    frame_duration_ms: int = 60,
    track_xy: tuple | None = None,
) -> go.Figure:
    """
    Build an animated Plotly figure for a single lap.

    Parameters
    ----------
    telemetry : dict from LapSimulator.build_telemetry()
    mini_sectors : raw YAML mini_sector rows (with 'turn' field)
    total_length_m : total lap distance [m]
    circuit_name : label for the figure title
    frame_step : subsample step for animation frames
    frame_duration_ms : milliseconds per frame
    track_xy : optional (s_arr, x_arr, y_arr) from FastF1 GPS data.
               If provided, uses real GPS coordinates instead of reconstructing.
    """
    # ── Track coordinates ─────────────────────────────────────────────────────
    if track_xy is not None:
        s_track, x_track, y_track = track_xy
    else:
        s_track, x_track, y_track = _reconstruct_track(mini_sectors, total_length_m)

    # ── Extract telemetry ────────────────────────────────────────────────────
    s_tele  = np.array(telemetry["s"])
    v_kmh   = np.array(telemetry["v_kmh"])
    thr     = np.array(telemetry["throttle"])
    brk     = np.array(telemetry["brake"])
    gear    = np.array(telemetry["gear"])

    # Subsampled indices for animation frames
    idx_frames = list(range(0, len(s_tele), frame_step))
    if idx_frames[-1] != len(s_tele) - 1:
        idx_frames.append(len(s_tele) - 1)

    # Interpolate car (x, y) for every telemetry point
    car_x = np.interp(s_tele, s_track, x_track)
    car_y = np.interp(s_tele, s_track, y_track)

    # ── Axis bounds ──────────────────────────────────────────────────────────
    pad = (x_track.max() - x_track.min()) * 0.06
    xmin, xmax = x_track.min() - pad, x_track.max() + pad
    ymin, ymax = y_track.min() - pad, y_track.max() + pad

    # ── Colour helpers ───────────────────────────────────────────────────────
    _TRACK_COL  = "#444444"
    _CAR_COL    = "#e8002d"
    _SPEED_COL  = "#4fc3f7"
    _TROT_COL   = "#66bb6a"
    _BRAKE_COL  = "#ef5350"
    _GEAR_COL   = "#ffd700"

    # ── Static traces ────────────────────────────────────────────────────────
    # 0: track outline
    trace_track = go.Scatter(
        x=x_track, y=y_track,
        mode="lines",
        line=dict(color=_TRACK_COL, width=8),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x", yaxis="y",
    )

    # 1: start/finish marker
    trace_sf = go.Scatter(
        x=[x_track[0]], y=[y_track[0]],
        mode="markers",
        marker=dict(symbol="line-ew", size=18, color="white",
                    line=dict(color="white", width=3)),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x", yaxis="y",
    )

    # 2: speed full-lap trace (background)
    trace_speed_bg = go.Scatter(
        x=s_tele, y=v_kmh,
        mode="lines",
        line=dict(color=_SPEED_COL, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.15)",
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2", yaxis="y2",
    )

    # 3: throttle trace (background)
    trace_thr_bg = go.Scatter(
        x=s_tele, y=thr * 100,
        mode="lines",
        line=dict(color=_TROT_COL, width=1),
        fill="tozeroy",
        fillcolor="rgba(102,187,106,0.2)",
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2", yaxis="y3",
    )

    # 4: brake trace (background)
    trace_brk_bg = go.Scatter(
        x=s_tele, y=brk * 100,
        mode="lines",
        line=dict(color=_BRAKE_COL, width=1),
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.2)",
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2", yaxis="y3",
    )

    # ── Animated traces (initial state = frame 0) ─────────────────────────────
    i0 = idx_frames[0]

    # 5: car dot
    trace_car = go.Scatter(
        x=[car_x[i0]], y=[car_y[i0]],
        mode="markers+text",
        marker=dict(size=14, color=_CAR_COL, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"  {v_kmh[i0]:.0f} km/h  G{gear[i0]}"],
        textposition="middle right",
        textfont=dict(color="white", size=12),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x", yaxis="y",
    )

    # 6: cursor line on speed panel (vertical line = 2 points same x)
    trace_cursor = go.Scatter(
        x=[s_tele[i0], s_tele[i0]],
        y=[0, v_kmh.max() * 1.05],
        mode="lines",
        line=dict(color=_CAR_COL, width=2, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
        xaxis="x2", yaxis="y2",
    )

    # 7: throttle bar (right panel) — single horizontal bar
    trace_thr_bar = go.Bar(
        x=[thr[i0] * 100],
        y=["T"],
        orientation="h",
        marker_color=_TROT_COL,
        showlegend=False,
        hoverinfo="skip",
        xaxis="x3", yaxis="y4",
    )

    # 8: brake bar (right panel)
    trace_brk_bar = go.Bar(
        x=[brk[i0] * 100],
        y=["B"],
        orientation="h",
        marker_color=_BRAKE_COL,
        showlegend=False,
        hoverinfo="skip",
        xaxis="x3", yaxis="y4",
    )

    # ── Build frames ─────────────────────────────────────────────────────────
    frames = []
    for fi, idx in enumerate(idx_frames):
        frames.append(go.Frame(
            name=str(fi),
            data=[
                # trace 5: car dot
                go.Scatter(
                    x=[car_x[idx]], y=[car_y[idx]],
                    text=[f"  {v_kmh[idx]:.0f} km/h  G{gear[idx]}"],
                ),
                # trace 6: cursor
                go.Scatter(
                    x=[s_tele[idx], s_tele[idx]],
                    y=[0, v_kmh.max() * 1.05],
                ),
                # trace 7: throttle bar
                go.Bar(x=[thr[idx] * 100], y=["T"]),
                # trace 8: brake bar
                go.Bar(x=[brk[idx] * 100], y=["B"]),
            ],
            traces=[5, 6, 7, 8],
        ))

    # ── Layout ───────────────────────────────────────────────────────────────
    layout = go.Layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{circuit_name}</b> — Lap Animation",
            x=0.5,
            font=dict(size=16),
        ),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        paper_bgcolor="#111",
        plot_bgcolor="#111",

        # ── Axis layout ───────────────────────────────────────────────────
        # Track map: top-left (65% width, top 60%)
        xaxis=dict(
            domain=[0.0, 0.65],
            anchor="y",
            showgrid=False, zeroline=False, showticklabels=False,
            range=[xmin, xmax],
        ),
        yaxis=dict(
            domain=[0.38, 1.0],
            anchor="x",
            showgrid=False, zeroline=False, showticklabels=False,
            range=[ymin, ymax],
            scaleanchor="x",
        ),

        # Speed trace: bottom (65% width, bottom 30%)
        xaxis2=dict(
            domain=[0.0, 0.65],
            anchor="y2",
            title=dict(text="Distance [m]", font=dict(size=10)),
            showgrid=True, gridcolor="#333",
            range=[0, total_length_m],
        ),
        yaxis2=dict(
            domain=[0.0, 0.32],
            anchor="x2",
            title=dict(text="Speed [km/h]", font=dict(size=10)),
            showgrid=True, gridcolor="#333",
            range=[0, v_kmh.max() * 1.1],
        ),

        # Throttle/Brake overlay on same panel (right y-axis)
        yaxis3=dict(
            domain=[0.0, 0.32],
            anchor="x2",
            overlaying="y2",
            side="right",
            range=[0, 1.1],
            showgrid=False,
            showticklabels=False,
        ),

        # Right panel bars: (35% width, top 60%)
        xaxis3=dict(
            domain=[0.70, 1.0],
            anchor="y4",
            range=[0, 100],
            showgrid=False,
            showticklabels=True,
            title=dict(text="[%]", font=dict(size=9)),
        ),
        yaxis4=dict(
            domain=[0.50, 0.80],
            anchor="x3",
            showgrid=False,
            tickfont=dict(size=13, color="white"),
        ),

        # ── Play / Pause buttons ──────────────────────────────────────────
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.68, y=0.45,
            xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": frame_duration_ms, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],

        # ── Lap slider ───────────────────────────────────────────────────
        sliders=[dict(
            active=0,
            x=0.0, y=-0.02,
            len=0.65,
            xanchor="left", yanchor="top",
            pad=dict(t=5, b=5),
            currentvalue=dict(
                prefix="Dist: ",
                suffix=" m",
                font=dict(size=11),
                visible=True,
                xanchor="left",
            ),
            transition=dict(duration=0),
            steps=[
                dict(
                    method="animate",
                    label=f"{s_tele[idx]:.0f}",
                    args=[[str(fi)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                )
                for fi, idx in enumerate(idx_frames)
            ],
        )],

        # ── Annotations (static labels) ───────────────────────────────────
        annotations=[
            dict(
                text="Speed (km/h)",
                x=0.68, y=0.98,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#aaa", size=10),
                xanchor="left",
            ),
            dict(
                text="Throttle / Brake",
                x=0.70, y=0.83,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#aaa", size=10),
                xanchor="left",
            ),
        ],
    )

    fig = go.Figure(
        data=[
            trace_track,    # 0 — static
            trace_sf,       # 1 — static
            trace_speed_bg, # 2 — static
            trace_thr_bg,   # 3 — static
            trace_brk_bg,   # 4 — static
            trace_car,      # 5 — animated
            trace_cursor,   # 6 — animated
            trace_thr_bar,  # 7 — animated
            trace_brk_bar,  # 8 — animated
        ],
        layout=layout,
        frames=frames,
    )

    return fig
