"""
F1 Race Strategy Engine — Streamlit web app.

A thin, interactive front-end over the existing engine: pick a circuit, then
explore the strategy ranking, Safety-Car robustness, an uncertain rain
forecast (Level C) and the multi-car undercut / overcut battle. All heavy
computation is cached so the controls stay responsive.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from main import build_track_from_yaml, build_vehicle
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.simulation.multi_car_simulator import MultiCarSimulator
from src.optimization.strategy_optimizer import DPStrategyOptimizer
from src.simulation.monte_carlo import MonteCarloRaceSimulator, SafetyCarParams
from src.simulation.weather_mc import WeatherMonteCarlo, build_wet_response
from src.models.weather import WeatherModel, WeatherForecast
from src.models.strategy import RaceStrategy, PitStop
from src.models.tyre import INTERMEDIATE, WET

TRACK_DIR = "data/tracks"
_ACCENT = "#16a34a"
# Distinct per-strategy palette (used for the Safety-Car risk bubbles, etc.).
_PALETTE = ["#16a34a", "#ef4444", "#3b82f6", "#f59e0b", "#a855f7",
            "#06b6d4", "#ec4899", "#84cc16"]

# Spatial integration step [m] for the strategy search. Fixed at the validated
# value: coarser steps skip the corner apex/braking constraints and produce
# physically wrong (optimistically fast) lap times, so this is NOT user-tunable.
STEP = 50.0


# ── Data helpers (cached) ─────────────────────────────────────────────────

def _circuits() -> dict[str, str]:
    out = {}
    for path in sorted(glob.glob(f"{TRACK_DIR}/*.yaml")):
        name = Path(path).stem.replace("_", " ").title()
        out[name] = path
    return out


@st.cache_resource(show_spinner=False)
def _load(yaml_path: str):
    track, loader = build_track_from_yaml(yaml_path)
    vehicle = build_vehicle(loader.vehicle_setup() or {})
    ri = loader.race_info()
    vehicle.fuel_mass = ri.fuel_load_kg
    vehicle.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    return track, loader, vehicle, ri


@st.cache_data(show_spinner=False)
def _optimize(yaml_path: str, step: float, max_stops: int, with_wets: bool):
    track, loader, vehicle, ri = _load(yaml_path)
    race_sim = RaceSimulator(LapSimulator(track, vehicle),
                             weather=loader.weather_model())
    compounds = list(loader.tyre_compounds().values())
    if with_wets:
        compounds = compounds + [INTERMEDIATE, WET]
    dp = DPStrategyOptimizer(race_simulator=race_sim, min_stint_laps=8, verbose=False)
    results = dp.optimize(
        num_laps=ri.race_laps, compounds=compounds, pit_loss=ri.pit_lane_delta_s,
        max_stops=max_stops,
        require_two_compounds=(loader.weather_model().max_wetness == 0.0 and not with_wets),
        step_size=step,
    )
    return results


@st.cache_data(show_spinner=False)
def _surface(yaml_path: str, step: float):
    track, loader, vehicle, ri = _load(yaml_path)
    comps = dict(loader.tyre_compounds())
    comps["Intermediate"], comps["Wet"] = INTERMEDIATE, WET
    return build_wet_response(track, vehicle, comps, step_size=step,
                              fuel_mass=ri.fuel_load_kg * 0.5)


@st.cache_data(show_spinner=False)
def _wet_candidates(yaml_path: str, max_stops: int):
    """Inter/wet-capable candidate strategies for the weather Monte Carlo.

    On a dry-nominal circuit the dry DP only proposes slick plans, so the
    forecast MC has nothing to gamble with. Here we optimise under a *fixed
    representative shower* (a mid-race wet window) to discover good
    intermediate strategies, then RE-SIMULATE them on the dry nominal so their
    per-lap baseline is consistent with the weather MC (which adds the wet
    delta from the dry reference). The shower's exact shape is only used to
    pick sensible inter pit windows; the forecast sliders then score them.
    """
    track, loader, _, ri = _load(yaml_path)
    vehicle = build_vehicle(loader.vehicle_setup() or {})
    vehicle.fuel_mass = ri.fuel_load_kg
    N = ri.race_laps
    onset, dur = max(2, N // 3), max(4, N // 3)
    kf = [(1, 0.0), (onset, 0.0), (onset + 2, 0.6),
          (onset + 2 + dur, 0.6), (onset + 4 + dur, 0.0), (N, 0.0)]
    wm = WeatherModel.from_keyframes([(min(l, N), w) for l, w in kf])
    comps = list(loader.tyre_compounds().values()) + [INTERMEDIATE, WET]
    wet_sim = RaceSimulator(LapSimulator(track, vehicle), weather=wm)
    dp = DPStrategyOptimizer(race_simulator=wet_sim, min_stint_laps=8, verbose=False)
    wet_results = dp.optimize(num_laps=N, compounds=comps, pit_loss=ri.pit_lane_delta_s,
                              max_stops=max_stops, require_two_compounds=False, step_size=STEP)
    # Re-simulate the top wet plans on the DRY nominal for a consistent baseline.
    dry_sim = RaceSimulator(LapSimulator(track, vehicle), weather=loader.weather_model())
    out = []
    for r in wet_results[:4]:
        strat = RaceStrategy(
            r.strategy, r.stints[0].compound,
            [PitStop(s.start_lap, s.compound, ri.pit_lane_delta_s) for s in r.stints[1:]])
        out.append(dry_sim.simulate(N, strat, step_size=STEP))
    return out


@st.cache_data(show_spinner=False)
def _baseline_telemetry(yaml_path: str):
    """One representative race-fuel lap (Soft) → telemetry channels for plotting."""
    track, loader, _, ri = _load(yaml_path)
    vehicle = build_vehicle(loader.vehicle_setup() or {})  # fresh (don't mutate cache)
    vehicle.fuel_mass = ri.fuel_load_kg
    sim = LapSimulator(track, vehicle, track_wetness=loader.weather_model().wetness(1))
    comp = loader.tyre_compounds().get("Soft") or list(loader.tyre_compounds().values())[0]
    res = sim.simulate(step_size=STEP, tyre_compound=comp, initial_fuel_mass=ri.fuel_load_kg)
    t = sim.build_telemetry(res["points"])
    return {k: np.asarray(t[k]) for k in ("s", "v_kmh", "throttle", "brake", "gear",
                                          "tyre_temperature", "grip_usage")}, res["total_time"]


@st.cache_data(show_spinner=False)
def _quali_compare(yaml_path: str):
    """Sim qualifying lap (low fuel + ERS) vs real FastF1 pole telemetry.

    Returns (sim, real, sim_lap_time) or raises — the caller guards for the
    network-dependent FastF1 fetch.
    """
    from src.data.fastf1_loader import get_qualifying_telemetry
    track, loader, _, ri = _load(yaml_path)
    fastf1_name = loader.fastf1_name()
    if not fastf1_name:
        raise RuntimeError("circuit has no FastF1 name")
    real = get_qualifying_telemetry(fastf1_name, ri.season)

    vehicle = build_vehicle(loader.vehicle_setup() or {})  # fresh
    vehicle.ers_power_kw = 120.0          # qualifying ERS mode
    sim = LapSimulator(track, vehicle, track_wetness=loader.weather_model().wetness(1))
    comp = loader.tyre_compounds().get("Soft") or list(loader.tyre_compounds().values())[0]
    warm = sim.simulate(step_size=STEP, tyre_compound=comp, initial_fuel_mass=3.0)
    warm2 = sim.simulate(step_size=STEP, tyre_compound=comp, initial_fuel_mass=3.0,
                         initial_speed=warm["final_speed"], initial_gear=warm["final_gear"])
    st_tel = sim.build_telemetry(warm2["points"])
    sim_out = {"s": np.asarray(st_tel["s"]), "v_kmh": np.asarray(st_tel["v_kmh"])}
    real_out = {"s": np.asarray(real["s"]), "v_kmh": np.asarray(real["speed_kmh"]),
                "driver": real["driver"], "lap_time_s": real["lap_time_s"]}
    return sim_out, real_out, warm2["total_time"]


# ── App ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="F1 Race Strategy Engine", page_icon="🏁", layout="wide")
st.title("🏁 F1 Race Strategy Engine")
st.caption("Physics-calibrated lap simulator + strategy decision engine under uncertainty. "
           "The physics is the engine; the product is the decision.")

circuits = _circuits()
with st.sidebar:
    st.header("Setup")
    circuit_name = st.selectbox("Circuit", list(circuits))
    yaml_path = circuits[circuit_name]
    max_stops = st.slider("Max pit stops", 1, 3, 2)
    st.markdown("---")
    st.caption("First run on a circuit builds the stint table (~1–2 min) at the "
               "validated 50 m integration step, then it is cached and the controls "
               "are instant.")

track, loader, vehicle, ri = _load(yaml_path)
weather = loader.weather_model()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Race laps", ri.race_laps)
c2.metric("Lap distance", f"{ri.lap_distance_km:.3f} km")
c3.metric("Pit loss", f"{ri.pit_lane_delta_s:.1f} s")
c4.metric("Overtaking ease", f"{loader.overtaking_likelihood():.2f}")

tab_strat, tab_sc, tab_weather, tab_multi, tab_tel, tab_val = st.tabs(
    ["📋 Strategy", "🟡 Safety-Car risk", "🌧 Weather forecast (Level C)",
     "🏎 Multi-car battle", "📈 Lap telemetry", "🆚 FastF1 vs Sim"])

# ── Strategy ranking ──────────────────────────────────────────────────────
with tab_strat:
    with st.spinner("Optimising strategies…"):
        results = _optimize(yaml_path, STEP, max_stops, with_wets=weather.max_wetness > 0)
    best = results[0]
    st.subheader(f"Optimal: {best.strategy}  ·  {best.total_time/60:.1f} min")
    rows = [{
        "Strategy": r.strategy, "Total [s]": round(r.total_time, 1),
        "Gap [s]": round(r.total_time - best.total_time, 1),
        "Stops": len(r.stints) - 1, "Avg raw [s]": round(r.average_raw_lap_time, 3),
    } for r in results[:10]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── Safety-Car Monte Carlo ────────────────────────────────────────────────
with tab_sc:
    st.caption("Sampling Safety Cars / VSCs from the circuit's historical rates: "
               "which plan is robust, not just fastest on paper?")
    with st.spinner("Optimising strategies…"):
        results = _optimize(yaml_path, STEP, max_stops, with_wets=weather.max_wetness > 0)
    sc_params = SafetyCarParams(
        sc_prob_per_lap=ri.sc_probability_per_lap,
        vsc_prob_per_lap=ri.vsc_probability_per_lap,
        avg_sc_duration_laps=ri.avg_sc_duration_laps,
    )
    dist = MonteCarloRaceSimulator(sc_params, num_samples=2000).evaluate(results[:6])
    neu = dist[0].neutralisation_pct
    st.metric("Races neutralised (SC or VSC)", f"{neu:.0f}%")
    fig = go.Figure()
    for i, d in enumerate(dist):
        fig.add_trace(go.Scatter(
            x=[d.p50], y=[d.p95], mode="markers+text",
            marker=dict(size=12 + d.win_probability * 60,
                        color=_PALETTE[i % len(_PALETTE)], opacity=0.75,
                        line=dict(width=1, color="#0d1117")),
            text=[f"{d.win_probability*100:.0f}%"], textposition="top center",
            name=d.name))
    fig.update_layout(
        xaxis_title="Median race time P50 [s] (lower = faster)",
        yaxis_title="Worst-case P95 [s] (lower = safer)",
        template="plotly_dark", height=430, showlegend=True,
        legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.caption("Bubble size = win probability. Lower-left = fast *and* safe.")

# ── Weather forecast (Level C) ────────────────────────────────────────────
with tab_weather:
    st.caption("An uncertain rain shower: how robust is each plan to the forecast "
               "being wrong? Set the forecast below, then press Run — this tab is "
               "opt-in because it also re-optimises with wet compounds and builds a "
               "wetness response surface (a few extra seconds).")
    w1, w2, w3, w4 = st.columns(4)
    p_rain = w1.slider("P(rain)", 0.0, 1.0, 0.6, 0.05)
    onset = w2.slider("Onset lap", 1, ri.race_laps, min(ri.race_laps // 2, 25))
    peak = w3.slider("Peak wetness", 0.1, 1.0, 0.6, 0.05)
    dur = w4.slider("Duration (laps)", 3, 30, 10)
    if st.button("Run weather Monte Carlo", type="primary"):
        with st.spinner("Optimising (dry + wet candidates) + building response surface…"):
            wres = _optimize(yaml_path, STEP, max_stops, with_wets=True)   # dry-optimal (slick)
            wet_cands = _wet_candidates(yaml_path, max_stops)              # inter/wet-capable
            surface = _surface(yaml_path, STEP)
        # Candidate pool: the dry-optimal slick plans + the inter-gamble plans,
        # deduplicated by name — so the forecast can shift the win% between them.
        pool, seen = [], set()
        for r in list(wres[:5]) + list(wet_cands):
            if r.strategy not in seen:
                seen.add(r.strategy); pool.append(r)
        forecast = WeatherForecast(
            rain_probability=p_rain, onset_lap_mean=onset, onset_lap_std=max(1, onset * 0.15),
            peak_wetness_mean=peak, peak_wetness_std=0.12, ramp_laps=2,
            duration_laps_mean=dur, duration_laps_std=max(1, dur * 0.3),
            race_laps=ri.race_laps)
        wd = WeatherMonteCarlo(forecast, surface, weather, num_samples=2000).evaluate(pool)
        st.metric("Sampled races that saw rain", f"{wd[0].rain_exposure_pct:.0f}%")
        wd_sorted = sorted(wd, key=lambda x: -x.win_probability)
        fig = go.Figure(go.Bar(
            x=[d.win_probability * 100 for d in wd_sorted],
            y=[d.name for d in wd_sorted], orientation="h",
            marker_color=_ACCENT))
        fig.update_layout(xaxis_title="Win % across forecast scenarios",
                          template="plotly_dark", height=430,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        det_best = min(wd, key=lambda x: x.deterministic_total)
        rob_best = max(wd, key=lambda x: x.win_probability)
        if det_best.name != rob_best.name:
            st.warning(f"The plan fastest under the nominal forecast (**{det_best.name}**) "
                       f"is **not** the most robust: **{rob_best.name}** wins "
                       f"{rob_best.win_probability*100:.0f}% of forecast scenarios.")
        else:
            st.success(f"**{rob_best.name}** is both fastest on the nominal forecast "
                       f"and the most robust to it being wrong.")

# ── Multi-car battle ──────────────────────────────────────────────────────
with tab_multi:
    st.caption("On-track passes are hard (scaled by the circuit's overtaking ease); "
               "position is won via the undercut / overcut.")
    with st.spinner("Optimising + racing…"):
        results = _optimize(yaml_path, STEP, max_stops, with_wets=weather.max_wetness > 0)
        race_sim = RaceSimulator(LapSimulator(track, vehicle), weather=weather)
        n = min(5, len(results))
        entries = []
        for i, r in enumerate(results[:n]):
            stints = r.stints
            pits = [PitStop(s.start_lap, s.compound, ri.pit_lane_delta_s) for s in stints[1:]]
            entries.append((f"Car {i+1} · {r.strategy}",
                            RaceStrategy(r.strategy, stints[0].compound, pits)))
        mc = MultiCarSimulator(race_sim, overtaking_likelihood=loader.overtaking_likelihood())
        mres = mc.simulate(entries, num_laps=ri.race_laps, step_size=STEP)
    fig = go.Figure()
    for car in mres.cars:
        fig.add_trace(go.Scatter(
            x=[lr.lap for lr in car.laps], y=[lr.position for lr in car.laps],
            mode="lines", name=car.name, line=dict(width=2.5)))
    fig.update_layout(xaxis_title="Lap", yaxis_title="Position",
                      yaxis=dict(autorange="reversed", dtick=1),
                      template="plotly_dark", height=430,
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    events = getattr(mres, "overtake_events", [])
    if events:
        ev_rows = [{"Lap": e.lap, "Gained by": e.gainer, "Over": e.loser,
                    "→ P": e.new_position, "Cause": e.kind} for e in events]
        st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
        n_u = sum(e.kind == "undercut" for e in events)
        n_o = sum(e.kind == "overcut" for e in events)
        n_t = sum(e.kind == "on-track pass" for e in events)
        st.caption(f"{len(events)} position changes — {n_u} undercut, {n_o} overcut, "
                   f"{n_t} on-track.")
    else:
        st.info("Grid order held to the flag (no position changes).")

# ── Lap telemetry ─────────────────────────────────────────────────────────
with tab_tel:
    st.caption("One representative race-fuel lap (Soft) — the physics under the "
               "strategy layer: speed, throttle/brake, gear and tyre state vs distance.")
    with st.spinner("Simulating lap…"):
        tel, lap_t = _baseline_telemetry(yaml_path)
    st.metric("Simulated lap time (race fuel, Soft)", f"{lap_t:.3f} s")
    s_km = tel["s"] / 1000.0
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Speed [km/h]", "Throttle / Brake [%]", "Gear", "Tyre temp [°C]"))
    fig.add_trace(go.Scatter(x=s_km, y=tel["v_kmh"], line=dict(color=_ACCENT),
                             name="Speed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_km, y=tel["throttle"], line=dict(color="#16a34a"),
                             name="Throttle"), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_km, y=tel["brake"], line=dict(color="#ef4444"),
                             name="Brake"), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_km, y=tel["gear"], line=dict(color="#3b82f6", shape="hv"),
                             name="Gear"), row=3, col=1)
    fig.add_trace(go.Scatter(x=s_km, y=tel["tyre_temperature"], line=dict(color="#f59e0b"),
                             name="Tyre temp"), row=4, col=1)
    fig.update_xaxes(title_text="Lap distance [km]", row=4, col=1)
    fig.update_layout(template="plotly_dark", height=720, showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ── FastF1 vs Sim (validation) ────────────────────────────────────────────
with tab_val:
    st.caption("Validation: the simulated qualifying lap (low fuel + ERS) overlaid on "
               "the real pole lap from FastF1. Needs internet on first run (then cached).")
    try:
        with st.spinner("Building qualifying sim + fetching real telemetry…"):
            sim_q, real_q, sim_lap = _quali_compare(yaml_path)
        real_lap = real_q["lap_time_s"]
        delta = sim_lap - real_lap
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated qualifying", f"{sim_lap:.3f} s")
        m2.metric(f"Real pole ({real_q['driver']})", f"{real_lap:.3f} s")
        m3.metric("Gap (sim − real)", f"{delta:+.3f} s")
        # Normalise both to % of lap so the traces share an x-axis (the sim uses a
        # stylised geometric track, so absolute corner positions differ slightly).
        sx = sim_q["s"] / sim_q["s"].max() * 100.0
        rx = real_q["s"] / real_q["s"].max() * 100.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rx, y=real_q["v_kmh"],
                                 name=f"Real ({real_q['driver']} pole)",
                                 line=dict(color="#ef4444", width=2)))
        fig.add_trace(go.Scatter(x=sx, y=sim_q["v_kmh"],
                                 name="Simulated", line=dict(color=_ACCENT, width=2, dash="dot")))
        fig.update_layout(xaxis_title="Lap distance [%]", yaxis_title="Speed [km/h]",
                          template="plotly_dark", height=460,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.caption("The lap-time gap is the calibration metric. The speed traces show the "
                   "sim reproduces the real top speeds and braking events; exact corner "
                   "positions differ because the sim track is a stylised geometric "
                   "reconstruction, not the GPS centreline. Note: Monza is *race*-calibrated, "
                   "so its lone-lap qualifying gap is larger by design (see README).")
    except Exception as e:
        st.warning(f"FastF1 comparison unavailable (needs network / cached data): {e}")
