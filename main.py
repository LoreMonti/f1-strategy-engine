# =========================================================
# F1 Lap Time Simulator and Race Strategy Optimizer
#
# Author: Lorenzo Monti
# Date: May 2026
# 
# Run:
# python main.py 
# =========================================================

import argparse
import glob
import os

from src.models.vehicle import Vehicle
from src.models.strategy import RaceResult
from src.models.tyre import SOFT, INTERMEDIATE, WET
from src.models.weather import WeatherModel
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.optimization.strategy_search import generate_and_simulate
from src.optimization.strategy_optimizer import DPStrategyOptimizer
from src.analysis.strategy_analysis import StrategyAnalyzer
from src.visualization.telemetry_plotter import TelemetryPlotter
from src.visualization.strategy_plotter import StrategyPlotter
from src.config import load_config
from src.data.loaders import TrackLoader

# ── Defaults (overridden by CLI) ──────────────────────────────────────────────
_DEFAULT_CIRCUIT    = "data/tracks/silverstone_2024.yaml"
_DEFAULT_SOLVER     = "dp"
_DEFAULT_STEP       = 50.0
_TRACKS_DIR         = "data/tracks"


def _resolve_circuit(value: str) -> str:
    """
    Accept a full path, a bare filename, or a short name (no extension / no path).
    Examples:
      monza_2024               → data/tracks/monza_2024.yaml
      monza_2024.yaml          → data/tracks/monza_2024.yaml  (if exists)
      data/tracks/monza.yaml   → unchanged
    """
    if os.path.isfile(value):
        return value
    # Try adding .yaml extension
    if os.path.isfile(value + ".yaml"):
        return value + ".yaml"
    # Try under the tracks directory
    candidate = os.path.join(_TRACKS_DIR, value)
    if os.path.isfile(candidate):
        return candidate
    candidate_yaml = candidate + ".yaml"
    if os.path.isfile(candidate_yaml):
        return candidate_yaml
    raise argparse.ArgumentTypeError(
        f"Circuit not found: '{value}'.\n"
        f"Available: {', '.join(sorted(glob.glob(os.path.join(_TRACKS_DIR, '*.yaml'))))}"
    )


def _parse_args() -> argparse.Namespace:
    available = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(_TRACKS_DIR, "*.yaml"))
    )

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="F1 Lap Time Simulator and Race Strategy Optimizer",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-c", "--circuit",
        metavar="CIRCUIT",
        default=_DEFAULT_CIRCUIT,
        help=(
            f"Circuit YAML path or short name.\n"
            f"Available: {', '.join(available)}\n"
            f"Default: {os.path.splitext(os.path.basename(_DEFAULT_CIRCUIT))[0]}"
        ),
    )
    parser.add_argument(
        "-s", "--solver",
        choices=["dp", "brute"],
        default=_DEFAULT_SOLVER,
        help=(
            "Strategy solver:\n"
            "  dp    — DPStrategyOptimizer (exact optimum, ~2-3 min)\n"
            "  brute — Brute-force sampled  (300 candidates, ~2-3 min)\n"
            f"Default: {_DEFAULT_SOLVER}"
        ),
    )
    parser.add_argument(
        "--step",
        type=float,
        default=_DEFAULT_STEP,
        metavar="METRES",
        help=(
            f"Step size for strategy search in metres (coarser = faster).\n"
            f"Telemetry/baseline always use 5 m for full fidelity.\n"
            f"Default: {_DEFAULT_STEP:.0f}"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Number of top strategies to display (default: from config).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print baseline lap and multi-lap tyre degradation table.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful for batch / headless runs).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Open interactive Plotly/Dash dashboard instead of saving PNG plots.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        metavar="PORT",
        help="Port for the Dash dashboard (default: 8050).",
    )
    parser.add_argument(
        "--multi-car",
        action="store_true",
        help="Run multi-car simulation using the top strategies as entries.",
    )
    parser.add_argument(
        "--num-cars",
        type=int,
        default=5,
        metavar="N",
        help="Number of cars for multi-car simulation (default: 5).",
    )
    parser.add_argument(
        "--wetness",
        type=float,
        default=None,
        metavar="0..1",
        help=(
            "Track wetness for the Level A weather model (0 = dry, 1 = soaked).\n"
            "Overrides the circuit YAML 'weather.track_wetness'. When > 0 the\n"
            "Intermediate and Wet compounds are added to the strategy search."
        ),
    )
    parser.add_argument(
        "--sc-lap",
        type=int,
        default=None,
        metavar="LAP",
        help=(
            "Live strategy demo: simulate a Safety Car deployed at this lap and\n"
            "print the optimal real-time pit decision (re-optimised from that\n"
            "race state), vs the green-flag plan."
        ),
    )
    parser.add_argument(
        "--weather-timeline",
        type=str,
        default=None,
        metavar="L:W,...",
        help=(
            "Level B dynamic weather: a comma-separated list of lap:wetness\n"
            "keyframes, interpolated per lap (e.g. '1:0,26:0,29:0.6,36:0.4,40:0').\n"
            "Overrides --wetness and the circuit YAML weather section."
        ),
    )

    return parser.parse_args()


def _parse_weather_timeline(spec: str) -> "WeatherModel":
    """Parse a 'lap:wetness,...' CLI string into a WeatherModel."""
    points = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        lap_s, wet_s = token.split(":")
        points.append((int(lap_s), float(wet_s)))
    return WeatherModel.from_keyframes(points)


# ------------------------------------------------------------------ #
# Builder functions                                                   #
# ------------------------------------------------------------------ #

def build_track_from_yaml(yaml_path: str = _DEFAULT_CIRCUIT):
    """Load a circuit from YAML and return (Track, TrackLoader)."""
    loader = TrackLoader(yaml_path)
    return loader.track(), loader


def build_vehicle(overrides: dict | None = None) -> Vehicle:
    """
    Build the base F1 vehicle, optionally applying circuit-specific overrides.

    ``overrides`` comes from ``loader.vehicle_setup()`` and can set any of:
    drag_coefficient, lift_coefficient, max_speed, fuel_mass, tyre_mu,
    max_brake_accel.  Everything else stays at the calibrated base values.
    """
    params: dict = dict(
        name="Generic F1 Car",
        mass=798.0,
        fuel_mass=105.0,               # default; overridden by circuit YAML
        fuel_consumption_per_km=0.342, # overridden in main() from race_info
        max_power=735_000.0,
        # ── ERS — off by default; overridden via circuit YAML vehicle_setup ───
        ers_power_kw=0.0,
        # ── Aero — Monza low-downforce baseline (overridden per circuit) ──────
        drag_coefficient=0.75,
        lift_coefficient=2.0,
        front_aero_balance=0.45,
        frontal_area=1.5,
        # ── Brakes ────────────────────────────────────────────────────────────
        max_brake_accel=5.5 * 9.81,    # 5.5g — calibrated to race pace
        # ── Tyres ─────────────────────────────────────────────────────────────
        tyre_mu=1.9,                   # calibrated to Monza 2024 race pace
        # ── Speed cap ─────────────────────────────────────────────────────────
        max_speed=95.0,                # 342 km/h Monza; overridden per circuit
        # ── Powertrain ────────────────────────────────────────────────────────
        gear_ratios=None,
        final_drive=3.5,
        wheel_radius=0.33,
        max_rpm=12_000.0,
        idle_rpm=4_000.0,
        peak_torque=820.0,
        drivetrain_efficiency=0.92,
    )
    if overrides:
        params.update(overrides)
    return Vehicle(**params)


# ------------------------------------------------------------------ #
# Print helpers                                                       #
# ------------------------------------------------------------------ #

def _header(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def print_project_summary(track, vehicle, lap_sim, step_size: float, race_info=None) -> None:
    label = f"{race_info.gp_name.upper()} {race_info.season}" if race_info else track.name.upper()
    print("=" * 50)
    print(f"F1 LAP TIME SIMULATOR — {label}")
    print("=" * 50)
    print(f"Track:      {track.name}  ({track.total_length:.0f} m, {track.num_segments} segments)")
    print(f"Vehicle:    {vehicle.name}")
    print(f"Step size:  {step_size:.1f} m")
    print(f"Simulator:  {lap_sim}")


def print_baseline_lap(lap_sim: LapSimulator, step_size: float) -> dict:
    _header("BASELINE LAP SIMULATION")

    result    = lap_sim.simulate(step_size=step_size, tyre_compound=SOFT)
    telemetry = lap_sim.build_telemetry(result["points"])

    print(f"Tyre compound:   {SOFT.name}")
    print(f"Lap time:        {result['total_time']:.3f} s")
    print(f"Max speed:       {telemetry['v_kmh'].max():.1f} km/h")
    print(f"Final tyre wear: {result['final_tyre_wear'] * 100:.1f} %")
    print(f"Final fuel mass: {result['final_fuel_mass']:.2f} kg")

    return {"result": result, "telemetry": telemetry}


def print_multi_lap_summary(lap_sim: LapSimulator, step_size: float, num_laps: int) -> dict:
    _header("MULTI-LAP SUMMARY")

    result = lap_sim.simulate_multiple_laps(
        num_laps=num_laps,
        step_size=step_size,
        tyre_compound=SOFT,
    )

    print(f"Tyre compound: {SOFT.name}")
    print(
        f"{'Lap':>3} | {'Time [s]':>8} | {'Delta [s]':>9} | "
        f"{'Wear [%]':>8} | {'Grip':>6} | {'Fuel [kg]':>9} | {'Vmax [km/h]':>11}"
    )
    print("-" * 78)

    for lap in result["laps"]:
        print(
            f"{lap['lap']:3d} | "
            f"{lap['lap_time']:8.3f} | "
            f"{lap['delta_lap_time']:9.3f} | "
            f"{lap['final_tyre_wear'] * 100:8.1f} | "
            f"{lap['final_grip_multiplier']:6.3f} | "
            f"{lap['final_fuel_mass']:9.2f} | "
            f"{lap['max_speed'] * 3.6:11.1f}"
        )

    return result


def print_strategy_ranking(race_results: list[RaceResult], top_n: int) -> None:
    _header("RACE STRATEGY RANKING")

    best_time = race_results[0].total_time
    print(f"Generated strategies: {len(race_results)}")
    print(
        f"{'Pos':>3} | {'Strategy':<28} | {'Total [s]':>10} | "
        f"{'Gap [s]':>8} | {'Stops':>5} | {'Fastest':>8} | {'Avg Raw':>8}"
    )
    print("-" * 84)

    for pos, result in enumerate(race_results[:top_n], start=1):
        print(
            f"{pos:3d} | "
            f"{result.strategy:<28} | "
            f"{result.total_time:10.3f} | "
            f"{result.total_time - best_time:8.3f} | "
            f"{result.num_stops:5d} | "
            f"{result.fastest_lap.raw_lap_time:8.3f} | "
            f"{result.average_raw_lap_time:8.3f}"
        )


def print_learned_degradation(learned_deg: dict, circuit: str) -> None:
    """Print the tyre-degradation rates learned from real FastF1 stints."""
    _header("LEARNED TYRE DEGRADATION (ML from real stints)")
    print(f"Source: FastF1 race stints at {circuit} (fuel-corrected, robust "
          f"Theil–Sen fit, empirical-Bayes shrinkage)")
    print(f"deg = degradation slope [s/lap], ±σ = spread across stints "
          f"(feeds the Monte Carlo degradation noise)")
    print(f"\n{'Compound':<14} | {'deg [s/lap]':>11} | {'±σ':>6} | "
          f"{'raw median':>10} | {'# stints':>8}")
    print("-" * 64)
    _order = {"Soft": 0, "Medium": 1, "Hard": 2, "Intermediate": 3, "Wet": 4}
    for comp, d in sorted(learned_deg.items(), key=lambda kv: _order.get(kv[0], 9)):
        print(
            f"{comp:<14} | {d.deg_s_per_lap:11.4f} | {d.deg_std:6.4f} | "
            f"{d.raw_deg_s_per_lap:10.4f} | {d.n_stints:8d}"
        )


def print_live_decision(reopt, best_result, sc_lap: int, sc_duration: int) -> None:
    """Demonstrate the live re-optimiser: optimal pit call under a Safety Car."""
    from src.optimization.live_reoptimizer import RaceState
    if sc_lap < 1 or sc_lap >= best_result.num_laps - 1:
        print(f"[Live] sc-lap {sc_lap} out of range; skipping live decision.")
        return
    state = RaceState.from_result(best_result, sc_lap)
    _header(f"LIVE STRATEGY DECISION — Safety Car deployed at lap {sc_lap}")
    print(f"Race state: on {state.compound.name} (age {state.tyre_age} laps), "
          f"compounds used so far: {', '.join(sorted(state.used_compounds))}")
    print(f"Re-optimising the remaining {best_result.num_laps - sc_lap} laps "
          f"(SC window ≈ {sc_duration} laps, pit discounted while it is out)...")

    g = reopt.decide(state, regime="green", max_remaining_stops=1)
    s = reopt.decide(state, regime="sc", sc_duration=sc_duration, max_remaining_stops=1)
    if not g or not s:
        print("  (no legal continuation found)")
        return
    gbest, sbest = g[0], s[0]
    print(f"\n  Green-flag optimum from here : {gbest.label}")
    print(f"  Optimum given the Safety Car : {sbest.label}")

    # Value of reacting: cost of sticking to the green plan vs the SC optimum,
    # both evaluated under the SAME Safety-Car scenario (common inflation cancels).
    gplan = [(p, c) for p, c in gbest.remaining_pits]
    match = next((o for o in s if [(p, c) for p, c in o.remaining_pits] == gplan), None)
    if sbest.label != gbest.label and match is not None:
        saving = match.remaining_time - sbest.remaining_time
        print(f"\n  → REACT: '{sbest.label}' saves {saving:.1f} s vs holding the "
              f"green plan under this Safety Car.")
    else:
        print(f"\n  → HOLD: the planned stop is already optimal; the Safety Car "
              f"does not change the call from this state.")


def print_monte_carlo(distributions, num_samples: int) -> None:
    """Print the Monte Carlo robustness table (strategies under SC/VSC)."""
    _header("MONTE CARLO — STRATEGY ROBUSTNESS UNDER SAFETY CARS")

    d0 = distributions[0] if distributions else None
    sc_pct  = d0.sc_exposure_pct if d0 else 0.0
    neu_pct = d0.neutralisation_pct if d0 else 0.0
    print(f"Samples: {num_samples}  |  {neu_pct:.0f}% of races had a neutralisation "
          f"({sc_pct:.0f}% a full Safety Car, the rest VSC-only)")
    print(f"P50 = median race time, spread = P95−P5 (smaller = more robust), "
          f"Win% = fraction of races won (paired sampling)")
    print(
        f"\n{'Strategy':<28} | {'Determin.':>10} | {'P50':>9} | "
        f"{'P5':>9} | {'P95':>9} | {'Spread':>7} | {'Win %':>6}"
    )
    print("-" * 96)
    # Rank by win probability (the robust pick), then median.
    for d in sorted(distributions, key=lambda x: (-x.win_probability, x.p50)):
        print(
            f"{d.name:<28} | "
            f"{d.deterministic_total:10.1f} | "
            f"{d.p50:9.1f} | {d.p5:9.1f} | {d.p95:9.1f} | "
            f"{d.robustness_s:7.1f} | {d.win_probability * 100:5.1f}%"
        )

    # Highlight the case where the deterministic best is NOT the robust best —
    # but only when it is a genuine trade-off, not two strategies that are
    # statistically tied on paper (within ~1 s the win% split is just noise).
    det_best = min(distributions, key=lambda x: x.deterministic_total)
    rob_best = max(distributions, key=lambda x: x.win_probability)
    paper_gap = rob_best.deterministic_total - det_best.deterministic_total
    if det_best.name != rob_best.name and paper_gap >= 1.0:
        print(
            f"\n  ⚠ The fastest strategy on paper ({det_best.name}) is NOT the most "
            f"robust:\n    '{rob_best.name}' is {paper_gap:.1f} s slower on paper but "
            f"wins {rob_best.win_probability * 100:.0f}% of races under Safety-Car "
            f"uncertainty (vs {det_best.win_probability * 100:.0f}%)."
        )


def print_decision_summary(best: RaceResult, distributions) -> None:
    """One-glance headline: fastest-on-paper vs most-robust strategy."""
    _header("RACE STRATEGY SUMMARY")
    print(f"Fastest on paper : {best.strategy}  ({best.total_time / 60:.1f} min)")
    if distributions:
        robust = max(distributions, key=lambda d: d.win_probability)
        print(f"Most robust (MC) : {robust.name}  "
              f"({robust.win_probability * 100:.0f}% of races won under Safety-Car risk)")
        gap = robust.deterministic_total - best.total_time
        if robust.name == best.strategy:
            print("Call             : same strategy is both fastest and most robust.")
        elif gap >= 1.0:
            print(f"Call             : trade-off — the robust pick is {gap:.1f} s slower "
                  f"on paper but safer to a Safety Car.")
        else:
            print("Call             : the top strategies are within ~1 s — effectively tied; "
                  "pick on tyre availability / track position.")


def print_best_strategy_details(best: RaceResult, show_laps: bool = False) -> None:
    _header("BEST STRATEGY DETAILS")

    print(f"Strategy:        {best.strategy}")
    print(f"Total race time: {best.total_time:.3f} s  "
          f"({best.total_time / 60:.1f} min)")
    print(f"Total pit loss:  {best.total_pit_loss:.3f} s")
    print(f"Fastest lap:     {best.fastest_lap.raw_lap_time:.3f} s")
    print(f"Avg raw lap:     {best.average_raw_lap_time:.3f} s")

    # The full lap-by-lap table is verbose; print it only with --verbose.
    if not show_laps:
        print("  (per-lap table: run with --verbose)")
        return

    print(
        f"\n{'Lap':>3} | {'Compound':>8} | {'Raw [s]':>8} | {'Pit [s]':>7} | "
        f"{'Total [s]':>9} | {'Cum [s]':>8} | {'Wear [%]':>8} | "
        f"{'Temp [°C]':>9} | {'Grip':>6}"
    )
    print("-" * 97)

    for lr in best.laps:
        marker = "*" if lr.pit_stop else " "
        print(
            f"{lr.lap:3d}{marker} | "
            f"{lr.compound:>8} | "
            f"{lr.raw_lap_time:8.3f} | "
            f"{lr.pit_time_loss:7.3f} | "
            f"{lr.lap_time:9.3f} | "
            f"{lr.cumulative_time:8.3f} | "
            f"{min(lr.final_tyre_wear, 1.0) * 100:8.1f} | "
            f"{lr.final_tyre_temperature:9.1f} | "
            f"{lr.final_grip_multiplier:6.3f}"
        )


def print_best_stint_summary(best: RaceResult) -> None:
    _header("BEST STRATEGY STINT SUMMARY")

    print(
        f"{'Stint':>5} | {'Compound':>8} | {'Start':>5} | {'End':>5} | "
        f"{'Laps':>5} | {'Raw [s]':>10} | {'Pit [s]':>9} | "
        f"{'Total [s]':>10} | {'Avg [s]':>8} | {'Wear [%]':>14}"
    )
    print("-" * 102)

    for i, stint in enumerate(best.stints, start=1):
        stint_laps = [lr for lr in best.laps if stint.contains_lap(lr.lap)]

        raw_time   = sum(lr.raw_lap_time  for lr in stint_laps)
        pit_loss   = sum(lr.pit_time_loss for lr in stint_laps)
        final_wear = min(stint_laps[-1].final_tyre_wear, 1.0) * 100.0

        print(
            f"{i:5d} | "
            f"{stint.compound.name:>8} | "
            f"{stint.start_lap:5d} | "
            f"{stint.end_lap:5d} | "
            f"{stint.length:5d} | "
            f"{raw_time:10.3f} | "
            f"{pit_loss:9.3f} | "
            f"{raw_time + pit_loss:10.3f} | "
            f"{raw_time / len(stint_laps):8.3f} | "
            f"{final_wear:14.1f}"
        )


# ------------------------------------------------------------------ #
# Multi-car print helpers                                            #
# ------------------------------------------------------------------ #

def print_multi_car_result(mc_result) -> None:
    from src.models.multi_car import MultiCarRaceResult

    _header("MULTI-CAR RACE RESULT")

    print(
        f"{'Pos':>3} | {'Car':<30} | {'Strategy':<30} | "
        f"{'Grid':>4} | {'Total [s]':>10} | {'Gap [s]':>8} | {'Traffic [s]':>11}"
    )
    print("-" * 110)

    winner_time = mc_result.cars[0].total_time
    for car in mc_result.cars:
        print(
            f"{car.final_position:3d} | "
            f"{car.name:<30} | "
            f"{car.strategy_name:<30} | "
            f"{car.grid_position:4d} | "
            f"{car.total_time:10.3f} | "
            f"{car.total_time - winner_time:8.3f} | "
            f"{car.total_traffic_penalty:11.3f}"
        )

    _header("MULTI-CAR POSITION EVOLUTION (every 5 laps)")
    cars = mc_result.cars
    header = f"{'Lap':>4} | " + " | ".join(f"{c.name[:12]:>12}" for c in cars)
    print(header)
    print("-" * len(header))

    sample_laps = list(range(1, mc_result.num_laps + 1, 5)) + [mc_result.num_laps]
    seen: set[int] = set()
    for lap_n in sample_laps:
        if lap_n in seen:
            continue
        seen.add(lap_n)
        # collect position of each car on this lap (sorted by final position)
        positions = {}
        for car in cars:
            lr = car.laps[lap_n - 1]
            positions[car.name] = lr.position
        row = f"{lap_n:4d} | " + " | ".join(f"P{positions[c.name]:>11}" for c in cars)
        print(row)

    # ── Undercut / overcut breakdown ──────────────────────────────────
    events = getattr(mc_result, "overtake_events", [])
    _header("POSITION CHANGES — UNDERCUT / OVERCUT / ON-TRACK")
    if not events:
        print("No position changes (grid order held to the flag).")
        return
    print(f"{'Lap':>4} | {'Gained by':<14} | {'Over':<14} | "
          f"{'→ P':>3} | {'Cause':<14}")
    print("-" * 62)
    for ev in events:
        print(
            f"{ev.lap:4d} | {ev.gainer:<14} | {ev.loser:<14} | "
            f"{ev.new_position:>3} | {ev.kind:<14}"
        )
    n_under = sum(1 for e in events if e.kind == "undercut")
    n_over = sum(1 for e in events if e.kind == "overcut")
    n_track = sum(1 for e in events if e.kind == "on-track pass")
    print(
        f"\nTotal: {len(events)} position changes  "
        f"({n_under} undercut, {n_over} overcut, {n_track} on-track)."
    )


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #

def main() -> None:
    # If no CLI arguments are given, launch the interactive TUI.
    # Any explicit flag bypasses the menu entirely (batch / scripting friendly).
    import sys
    if len(sys.argv) == 1:
        from src.tui import run_interactive
        args = run_interactive()
    else:
        args = _parse_args()

    circuit_yaml       = _resolve_circuit(args.circuit)
    use_dp             = args.solver == "dp"
    strategy_step_size = args.step
    verbose_physics    = args.verbose
    show_plots         = not args.no_plots and not args.dashboard
    show_dashboard     = args.dashboard
    run_multi_car      = args.multi_car
    num_cars           = args.num_cars

    # --- Config ---------------------------------------------------------
    sim_cfg, strat_cfg = load_config()

    top_n     = args.top_n if args.top_n is not None else strat_cfg.ranking_top_n
    gap_top_n = args.top_n if args.top_n is not None else strat_cfg.gap_top_n

    # --- Build objects --------------------------------------------------
    track, loader = build_track_from_yaml(circuit_yaml)
    vehicle = build_vehicle(loader.vehicle_setup())

    race_info = loader.race_info()
    race_laps = race_info.race_laps
    pit_loss  = race_info.pit_lane_delta_s

    vehicle.fuel_mass = race_info.fuel_load_kg
    vehicle.fuel_consumption_per_km = (
        race_info.fuel_consumption_kg_per_lap / race_info.lap_distance_km
    )

    # --- Weather model (Level A static / Level B dynamic) ---------------
    # A CLI --wetness > 0 forces a constant (Level A) wetness, overriding the
    # YAML. Otherwise the circuit's weather model is used: a 'weather.timeline'
    # (Level B dynamic) if present, else the static 'weather.track_wetness'.
    # Priority: --weather-timeline (CLI dynamic) > --wetness (CLI static,
    # including 0 to force a dry counterfactual against a wet YAML) > the
    # circuit YAML weather model (timeline if present, else static, else dry).
    _timeline_spec = getattr(args, "weather_timeline", None)
    if _timeline_spec:
        weather = _parse_weather_timeline(_timeline_spec)
    elif args.wetness is not None:
        weather = WeatherModel.constant(args.wetness)
    else:
        weather = loader.weather_model()

    max_wetness = weather.max_wetness

    compounds = list(loader.tyre_compounds().values())
    # When the track is (or becomes) wet, add the crossover/full-wet compounds
    # so the strategy search can choose them; slicks stay available (they still
    # win in a light damp / on a drying track).
    if max_wetness > 0.0:
        compounds = compounds + [INTERMEDIATE, WET]

    lap_sim  = LapSimulator(track, vehicle, track_wetness=weather.wetness(1))
    race_sim = RaceSimulator(lap_sim, weather=weather)

    print_project_summary(track, vehicle, lap_sim, sim_cfg.step_size, race_info)
    print(f"  Race laps   : {race_laps}  (from {race_info.circuit_id} YAML)")
    print(f"  Pit delta   : {pit_loss} s  (from {race_info.circuit_id} YAML)")
    print(f"  Compounds   : {[c.name for c in compounds]}")
    if max_wetness > 0.0:
        print(f"  Weather     : {weather.summary()}")
    print(f"  Solver      : {'DP optimizer' if use_dp else 'Brute-force (sampled)'}")
    print(f"  Search step : {strategy_step_size:.0f} m  "
          f"(telemetry/baseline: {sim_cfg.step_size:.0f} m)")

    # --- Baseline & multi-lap -------------------------------------------
    # Baseline lap is printed only with --verbose; data is always computed
    # for the dashboard/plots. Multi-lap summary is never printed (available
    # in the dashboard).
    if verbose_physics:
        baseline = print_baseline_lap(lap_sim, sim_cfg.step_size)
    else:
        baseline = {
            'result':    lap_sim.simulate(step_size=sim_cfg.step_size, tyre_compound=compounds[0]),
            'telemetry': lap_sim.build_telemetry(
                lap_sim.simulate(step_size=sim_cfg.step_size, tyre_compound=compounds[0])['points']
            ),
        }

    multi_lap_res = lap_sim.simulate_multiple_laps(
        num_laps=sim_cfg.multi_lap_laps,
        step_size=sim_cfg.step_size,
        tyre_compound=compounds[0],
    )

    # --- Strategy pool --------------------------------------------------
    # The mandatory two-compound rule is a DRY-race regulation; it is waived
    # in wet/declared-wet conditions, where a single wet-weather compound may
    # be run for the whole race (no forced second compound / extra stop).
    require_two = strat_cfg.require_two_compounds and max_wetness == 0.0

    if use_dp:
        dp = DPStrategyOptimizer(
            race_simulator=race_sim,
            min_stint_laps=strat_cfg.min_stint_laps,
            verbose=True,
        )
        race_results = dp.optimize(
            num_laps=race_laps,
            compounds=compounds,
            pit_loss=pit_loss,
            max_stops=strat_cfg.max_stops,
            require_two_compounds=require_two,
            step_size=strategy_step_size,
        )
    else:
        race_results = generate_and_simulate(
            race_simulator=race_sim,
            num_laps=race_laps,
            compounds=compounds,
            pit_loss=pit_loss,
            min_stint_laps=strat_cfg.min_stint_laps,
            max_stops=strat_cfg.max_stops,
            require_two_compounds=require_two,
            step_size=strategy_step_size,
            max_candidates=300,
            verbose=True,
        )

    print_strategy_ranking(race_results, top_n)

    # --- Monte Carlo robustness (SC/VSC under uncertainty) --------------
    # Turns the deterministic ranking into risk distributions: the fastest
    # strategy on paper is not always the most robust to a Safety Car.
    from src.simulation.monte_carlo import MonteCarloRaceSimulator, SafetyCarParams
    _mc_n = 2000

    # SC/VSC probabilities: estimated from real race-control history
    # (FastF1 TrackStatus, 2019→season, empirical-Bayes shrinkage) when
    # available; the YAML values are only a fallback for offline runs.
    sc_est = None
    _ff1_name = loader.fastf1_name()
    if _ff1_name:
        try:
            from src.data.sc_history import estimate_sc_params
            sc_est = estimate_sc_params(
                _ff1_name, list(range(2019, race_info.season + 1)))
        except Exception as e:
            print(f"[SC history] estimation unavailable ({e}); using YAML values")

    if sc_est is not None:
        sc_params = SafetyCarParams(
            sc_prob_per_lap=sc_est.sc_prob_per_lap,
            vsc_prob_per_lap=sc_est.vsc_prob_per_lap,
            avg_sc_duration_laps=max(1, round(sc_est.avg_sc_duration_laps)),
            avg_vsc_duration_laps=max(1, round(sc_est.avg_vsc_duration_laps)),
        )
        print(f"[SC history] {sc_est.circuit} {sc_est.years_used}: "
              f"{sc_est.sc_deployments} SC + {sc_est.vsc_deployments} VSC in "
              f"{sc_est.races_used} races → sc/lap={sc_est.sc_prob_per_lap:.4f}, "
              f"vsc/lap={sc_est.vsc_prob_per_lap:.4f} (YAML fallback: "
              f"{race_info.sc_probability_per_lap}/{race_info.vsc_probability_per_lap})")
    else:
        sc_params = SafetyCarParams(
            sc_prob_per_lap=race_info.sc_probability_per_lap,
            vsc_prob_per_lap=race_info.vsc_probability_per_lap,
            avg_sc_duration_laps=race_info.avg_sc_duration_laps,
        )
    # --- Learned tyre degradation (ML from real stints) -----------------
    # Degradation slope per compound, fitted to real FastF1 stints with
    # uncertainty. Surfaced here and fed to the Monte Carlo as deg noise.
    deg_uncertainty = None
    if _ff1_name:
        try:
            from src.data.tyre_deg import learn_degradation
            fuel_spl = race_info.fuel_effect_s_per_kg * race_info.fuel_consumption_kg_per_lap
            learned_deg = learn_degradation(
                _ff1_name, list(range(2019, race_info.season + 1)), fuel_spl)
            print_learned_degradation(learned_deg, _ff1_name)
            deg_uncertainty = {c: d.deg_std for c, d in learned_deg.items()}
        except Exception as e:
            print(f"[Tyre ML] degradation model unavailable ({e})")

    mc_strategies = race_results[:max(top_n, 5)]
    mc_dist = MonteCarloRaceSimulator(sc_params, num_samples=_mc_n).evaluate(
        mc_strategies, deg_uncertainty=deg_uncertainty)
    print_monte_carlo(mc_dist, _mc_n)

    # --- Live re-optimisation demo (optional, --sc-lap) -----------------
    if getattr(args, "sc_lap", None) is not None:
        from src.optimization.live_reoptimizer import LiveReoptimizer
        reopt = LiveReoptimizer(
            lap_sim=lap_sim, num_laps=race_laps, pit_loss=pit_loss,
            compounds=compounds, min_stint_laps=strat_cfg.min_stint_laps,
            require_two_compounds=require_two, step_size=strategy_step_size,
            weather=weather,
        )
        print_live_decision(reopt, race_results[0], args.sc_lap,
                            sc_params.avg_sc_duration_laps)

    # --- Analysis -------------------------------------------------------
    analyzer = StrategyAnalyzer(race_results)
    # Virtual gap evolution and undercut analysis available in dashboard only.

    best = analyzer.best_result
    print_best_strategy_details(best, show_laps=verbose_physics)
    print_best_stint_summary(best)

    # --- Multi-car simulation -------------------------------------------
    mc_result = None
    if run_multi_car:
        from src.simulation.multi_car_simulator import MultiCarSimulator
        from src.models.strategy import RaceStrategy, PitStop

        def _rebuild_strategy(rr: RaceResult) -> RaceStrategy:
            """Reconstruct a RaceStrategy from a RaceResult's stint list."""
            stints = rr.stints
            pit_stops = []
            for stint in stints[1:]:   # all stints except the first need a pit stop
                pit_stops.append(PitStop(
                    lap=stint.start_lap,
                    new_compound=stint.compound,
                    time_loss=pit_loss,
                ))
            return RaceStrategy(
                name=rr.strategy,
                initial_compound=stints[0].compound,
                pit_stops=pit_stops,
            )

        n = min(num_cars, len(race_results))
        entries = []
        for i, rr in enumerate(race_results[:n]):
            strat = _rebuild_strategy(rr)
            entries.append((f"Car {i+1}", strat))

        mc_sim = MultiCarSimulator(
            race_sim,
            overtaking_likelihood=loader.overtaking_likelihood(),
        )
        print(
            f"  Overtaking likelihood: {loader.overtaking_likelihood():.2f}  "
            f"→ pass margin {mc_sim.overtake_margin_s:.2f} s/lap "
            f"(higher margin = harder to pass, undercut/overcut matter more)"
        )
        mc_result = mc_sim.simulate(
            entries=entries,
            num_laps=race_laps,
            step_size=strategy_step_size,
        )
        print_multi_car_result(mc_result)

    # --- Headline decision summary --------------------------------------
    print_decision_summary(best, mc_dist)

    # --- Plots ----------------------------------------------------------
    if show_plots:
        _header("PLOT GENERATION")
        TelemetryPlotter.plot_telemetry(baseline["telemetry"])
        TelemetryPlotter.plot_multi_lap_summary(multi_lap_res)
        StrategyPlotter.plot_strategy_gaps(race_results, top_n=gap_top_n)
        StrategyPlotter.plot_strategy_lap_times(race_results, top_n=gap_top_n)
        StrategyPlotter.plot_strategy_tyre_wear(race_results, top_n=gap_top_n)
        StrategyPlotter.plot_strategy_temperature(race_results, top_n=gap_top_n)
    elif not show_dashboard:
        print("\n[plots skipped — pass without --no-plots to generate them]")

    if show_dashboard:
        from src.visualization.dashboard import launch
        from src.visualization.track_animator import build_lap_animation

        circuit_label = race_info.gp_name + " " + str(race_info.season)

        # Try to get real GPS track coordinates from FastF1
        track_xy = None
        fastf1_name = loader.fastf1_name()
        if fastf1_name:
            try:
                from src.data.fastf1_loader import get_track_map
                track_xy = get_track_map(fastf1_name, race_info.season)
            except Exception as e:
                print(f"[FastF1] Could not load track map: {e}")
                print("[FastF1] Falling back to geometric reconstruction.")

        animation_fig = build_lap_animation(
            telemetry=baseline["telemetry"],
            mini_sectors=loader.mini_sectors_raw(),
            total_length_m=race_info.lap_distance_km * 1000,
            circuit_name=circuit_label,
            track_xy=track_xy,
        )

        # FastF1 comparison data
        f1_comparison = None
        if fastf1_name:
            try:
                from src.data.fastf1_loader import (
                    get_qualifying_telemetry, get_race_laps,
                )
                from src.visualization.fastf1_comparison import (
                    fig_telemetry_comparison, fig_race_laps_comparison,
                )
                real_tel   = get_qualifying_telemetry(fastf1_name, race_info.season)
                real_laps  = get_race_laps(fastf1_name, race_info.season)

                # Warm-start qualifying sim:
                #   - Low fuel (~3 kg, matching a real flying lap)
                #   - Full ERS deployment (120 kW MGU-K, qualifying mode)
                # Both are restored after the comparison is built.
                _QUALI_FUEL_KG  = 3.0
                _QUALI_ERS_KW   = 120.0
                _race_ers_kw    = lap_sim.vehicle.ers_power_kw   # save race value
                lap_sim.vehicle.ers_power_kw = _QUALI_ERS_KW
                # Deterministic qualifying conditions: use the lap-1 wetness
                # (the race loop may have left track_wetness at the final lap).
                lap_sim.track_wetness = weather.wetness(1)
                lap_sim._track_points_cache.clear()               # aero cache stale

                _warm = lap_sim.simulate(
                    step_size=sim_cfg.step_size,
                    tyre_compound=SOFT,
                    initial_fuel_mass=_QUALI_FUEL_KG,
                )
                _warm2 = lap_sim.simulate(
                    step_size=sim_cfg.step_size,
                    tyre_compound=SOFT,
                    initial_speed=_warm["final_speed"],
                    initial_gear=_warm["final_gear"],
                    initial_fuel_mass=_QUALI_FUEL_KG,
                )
                sim_tel = lap_sim.build_telemetry(_warm2["points"])
                sim_tel["lap_time_s"] = _warm2["total_time"]

                # Restore race ERS and rebuild the speed-limit cache
                lap_sim.vehicle.ers_power_kw = _race_ers_kw
                lap_sim._track_points_cache.clear()

                best_result = analyzer.best_result

                f1_comparison = {
                    "tel_fig":  fig_telemetry_comparison(
                        sim_tel, real_tel, circuit_label),
                    "race_fig": fig_race_laps_comparison(
                        best_result, real_laps, circuit_label,
                        sim_wet=(max_wetness > 0.0)),
                }
                if real_laps.get("wet_race"):
                    if max_wetness > 0.0:
                        print("[FastF1] 🌧 WET/MIXED RACE — sim weather model active "
                              "(comparison meaningful)")
                    else:
                        print("[FastF1] ⚠ WET/MIXED RACE detected — race comparison "
                              "not meaningful (sim is dry; use --weather-timeline)")
                print(f"[FastF1] Comparison ready — "
                      f"real avg {real_laps['avg_lap_s']:.3f}s  "
                      f"sim avg {best_result.average_raw_lap_time:.3f}s  "
                      f"gap {best_result.average_raw_lap_time - real_laps['avg_lap_s']:+.3f}s/lap")
            except Exception as e:
                print(f"[FastF1] Comparison unavailable: {e}")

        launch(
            race_results=race_results,
            telemetry=baseline["telemetry"],
            multi_lap_res=multi_lap_res,
            circuit_name=circuit_label,
            port=args.port,
            mc_result=mc_result,
            animation_fig=animation_fig,
            f1_comparison=f1_comparison,
            mc_distributions=mc_dist,
        )
        return  # dashboard blocks; no "completed" message needed

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
    
