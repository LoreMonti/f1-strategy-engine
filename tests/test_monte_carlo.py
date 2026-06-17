"""Tests for the Monte Carlo race simulator (SC/VSC robustness)."""
import numpy as np
import pytest

from main import build_vehicle, build_track_from_yaml
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.simulation.monte_carlo import MonteCarloRaceSimulator, SafetyCarParams
from src.models.strategy import RaceStrategy, PitStop


@pytest.fixture(scope="module")
def small_races():
    """Two short Monza races (different strategies) — fast, no network."""
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    ri = loader.race_info()
    veh = build_vehicle(loader.vehicle_setup() or {})
    veh.fuel_mass = ri.fuel_load_kg
    veh.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    rs = RaceSimulator(LapSimulator(track, veh))
    comps = loader.tyre_compounds()
    P = ri.pit_lane_delta_s
    N = 14
    a = rs.simulate(N, RaceStrategy("A", comps["Medium"], [PitStop(7, comps["Soft"], P)]), step_size=50.0)
    b = rs.simulate(N, RaceStrategy("B", comps["Soft"], [PitStop(7, comps["Medium"], P)]), step_size=50.0)
    return [a, b]


@pytest.fixture(scope="module")
def sc_params():
    return SafetyCarParams(sc_prob_per_lap=0.02, vsc_prob_per_lap=0.015,
                           avg_sc_duration_laps=4)


def test_win_probabilities_sum_to_one(small_races, sc_params):
    dist = MonteCarloRaceSimulator(sc_params, num_samples=1000, seed=1).evaluate(small_races)
    total = sum(d.win_probability for d in dist)
    assert abs(total - 1.0) < 1e-6


def test_percentiles_are_ordered(small_races, sc_params):
    dist = MonteCarloRaceSimulator(sc_params, num_samples=1000, seed=2).evaluate(small_races)
    for d in dist:
        assert d.p5 <= d.p50 <= d.p95


def test_neutralisation_pct_is_a_percentage(small_races, sc_params):
    dist = MonteCarloRaceSimulator(sc_params, num_samples=1000, seed=3).evaluate(small_races)
    assert 0.0 <= dist[0].sc_exposure_pct <= dist[0].neutralisation_pct <= 100.0


def test_degradation_noise_is_the_only_spread_when_events_off(small_races):
    # Controlled comparison: no Safety Cars, no pace noise → with no
    # degradation uncertainty every race is identical (std ≈ 0); adding
    # degradation uncertainty is then the ONLY source of spread (std > 0).
    quiet = SafetyCarParams(sc_prob_per_lap=0.0, vsc_prob_per_lap=0.0,
                            avg_sc_duration_laps=4, pace_noise_s=0.0)
    mc = MonteCarloRaceSimulator(quiet, num_samples=1500, seed=4)
    base = mc.evaluate(small_races)
    noisy = mc.evaluate(small_races, deg_uncertainty={"Medium": 0.05, "Soft": 0.05})
    assert base[0].std < 1e-6
    assert noisy[0].std > 1.0


def test_clean_only_scenarios_match_deterministic():
    """With zero event probability and zero pace noise, totals == deterministic."""
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    ri = loader.race_info()
    veh = build_vehicle(loader.vehicle_setup() or {})
    veh.fuel_mass = ri.fuel_load_kg
    veh.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    rs = RaceSimulator(LapSimulator(track, veh))
    comps = loader.tyre_compounds()
    race = rs.simulate(12, RaceStrategy("A", comps["Medium"],
                       [PitStop(6, comps["Soft"], ri.pit_lane_delta_s)]), step_size=50.0)
    params = SafetyCarParams(sc_prob_per_lap=0.0, vsc_prob_per_lap=0.0,
                             avg_sc_duration_laps=4, pace_noise_s=0.0)
    dist = MonteCarloRaceSimulator(params, num_samples=50, seed=0).evaluate([race])
    assert abs(dist[0].p50 - race.total_time) < 1e-6
    assert dist[0].neutralisation_pct == 0.0
