"""
Generate docs/race_battle.gif — an animated multi-car position battle that
illustrates the project's thesis: strategy (pit timing + compound choice)
decides track position through the undercut / overcut, not raw pace alone.

Run from the repo root:  python scripts/make_readme_gif.py
No network needed (uses the local Monza circuit YAML).
"""
import logging; logging.disable(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from main import build_vehicle, build_track_from_yaml
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.simulation.multi_car_simulator import MultiCarSimulator
from src.models.strategy import RaceStrategy, PitStop

OUT = Path("docs/race_battle.gif")

# Brand-ish palette (F1 night-race dark theme).
BG     = "#0d1117"
FG     = "#e6edf3"
GRID   = "#30363d"
COLORS = ["#16a34a", "#ef4444", "#3b82f6", "#f59e0b", "#a855f7"]


def _build_race():
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    ri = loader.race_info()
    veh = build_vehicle(loader.vehicle_setup() or {})
    veh.fuel_mass = ri.fuel_load_kg
    veh.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    rs = RaceSimulator(LapSimulator(track, veh))
    comps = loader.tyre_compounds()
    P = ri.pit_lane_delta_s
    N = ri.race_laps
    # Five distinct strategies so the pit cycles cross over.
    entries = [
        ("Medium → Soft (L32)", RaceStrategy("M-S L32", comps["Medium"], [PitStop(32, comps["Soft"], P)])),
        ("Soft → Medium (L20)", RaceStrategy("S-M L20", comps["Soft"], [PitStop(20, comps["Medium"], P)])),
        ("Hard → Medium (L18)", RaceStrategy("H-M L18", comps["Hard"], [PitStop(18, comps["Medium"], P)])),
        ("Soft → Hard (L26)",   RaceStrategy("S-H L26", comps["Soft"], [PitStop(26, comps["Hard"], P)])),
        ("Medium → Hard (L34)", RaceStrategy("M-H L34", comps["Medium"], [PitStop(34, comps["Hard"], P)])),
    ]
    sim = MultiCarSimulator(rs, overtaking_likelihood=loader.overtaking_likelihood())
    result = sim.simulate(entries, num_laps=N, step_size=120.0)
    return result, [e[0] for e in entries], N


def main():
    result, labels, N = _build_race()
    ncars = len(result.cars)

    # positions[name][lap] and pit laps
    pos = {c.name: np.array([lr.position for lr in c.laps]) for c in result.cars}
    pit = {c.name: {lr.lap for lr in c.laps if lr.pit_stop} for c in result.cars}
    name_label = {c.name: lab for c, lab in zip(
        sorted(result.cars, key=lambda c: c.grid_position), labels)}
    # Map name → color by grid order for stable colours.
    order = [c.name for c in sorted(result.cars, key=lambda c: c.grid_position)]
    color = {nm: COLORS[i % len(COLORS)] for i, nm in enumerate(order)}

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=110)
    ax.set_xlim(1, N)
    ax.set_ylim(ncars + 0.5, 0.5)            # P1 at top
    ax.set_yticks(range(1, ncars + 1))
    ax.set_yticklabels([f"P{p}" for p in range(1, ncars + 1)])
    ax.set_xlabel("Lap")
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    title = ax.set_title("", color=FG, fontsize=13, weight="bold", loc="left")

    laps_axis = np.arange(1, N + 1)
    lines, dots, pit_marks = {}, {}, {}
    for nm in order:
        (ln,) = ax.plot([], [], color=color[nm], lw=2.4, alpha=0.95, solid_capstyle="round")
        (dot,) = ax.plot([], [], "o", color=color[nm], ms=9,
                         mec=BG, mew=1.5, zorder=5)
        lines[nm] = ln
        dots[nm] = dot
        pit_marks[nm] = ax.scatter([], [], marker="s", s=55, color=color[nm],
                                   edgecolors=FG, linewidths=0.8, zorder=6)

    # Leader-label text handles
    labels_txt = {nm: ax.text(0, 0, "", color=color[nm], fontsize=9,
                              va="center", ha="left", weight="bold")
                  for nm in order}

    def frame(k):  # k = lap index 0..N-1
        lap = k + 1
        for nm in order:
            y = pos[nm][:lap]
            x = laps_axis[:lap]
            lines[nm].set_data(x, y)
            dots[nm].set_data([lap], [pos[nm][lap - 1]])
            pls = sorted(p for p in pit[nm] if p <= lap)
            if pls:
                pit_marks[nm].set_offsets(np.c_[pls, [pos[nm][p - 1] for p in pls]])
            # label just right of the dot
            labels_txt[nm].set_position((lap + 0.4, pos[nm][lap - 1]))
            labels_txt[nm].set_text(name_label[nm] if lap >= N - 2 else "")
        title.set_text(f"Monza · multi-car strategy battle · Lap {lap}/{N}")
        return list(lines.values()) + list(dots.values())

    anim = FuncAnimation(fig, frame, frames=N, interval=140, blit=False)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {N} frames)")


if __name__ == "__main__":
    main()
