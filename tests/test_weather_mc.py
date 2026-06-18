"""Tests for the Level-C weather forecast and weather Monte Carlo."""
import numpy as np
import pytest

from main import build_vehicle, build_track_from_yaml
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.models.strategy import RaceStrategy, PitStop
from src.models.weather import WeatherModel, WeatherForecast
from src.simulation.weather_mc import WeatherMonteCarlo, build_wet_response


# ── WeatherForecast sampling ──────────────────────────────────────────────

def test_forecast_stays_dry_when_rain_does_not_fall():
    fc = WeatherForecast(rain_probability=0.0, onset_lap_mean=10, onset_lap_std=2,
                         peak_wetness_mean=0.6, peak_wetness_std=0.1, ramp_laps=2,
                         duration_laps_mean=8, duration_laps_std=2, race_laps=30)
    rng = np.random.default_rng(0)
    wm = fc.sample(rng)
    assert wm.max_wetness == 0.0


def test_forecast_produces_rain_when_certain():
    fc = WeatherForecast(rain_probability=1.0, onset_lap_mean=10, onset_lap_std=2,
                         peak_wetness_mean=0.6, peak_wetness_std=0.1, ramp_laps=2,
                         duration_laps_mean=8, duration_laps_std=2, race_laps=30)
    rng = np.random.default_rng(1)
    wm = fc.sample(rng)
    assert wm.max_wetness > 0.0
    assert wm.is_dynamic


def test_forecast_respects_race_length():
    fc = WeatherForecast(rain_probability=1.0, onset_lap_mean=28, onset_lap_std=1,
                         peak_wetness_mean=0.6, peak_wetness_std=0.1, ramp_laps=3,
                         duration_laps_mean=10, duration_laps_std=1, race_laps=30)
    rng = np.random.default_rng(2)
    wm = fc.sample(rng)
    assert all(1 <= lap <= 30 for lap, _ in wm.keyframes)


# ── Response surface ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def monza_surface():
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    veh = build_vehicle(loader.vehicle_setup() or {})
    comps = loader.tyre_compounds()
    buckets = np.array([0.0, 0.4, 0.8])
    surf = build_wet_response(track, veh, comps, buckets=buckets,
                              step_size=120.0, fuel_mass=80.0)
    return surf, buckets, track, veh, loader


def test_slick_lap_time_rises_with_wetness(monza_surface):
    surf, buckets, *_ = monza_surface
    soft = surf["Soft"]
    # A slick gets monotonically slower as the track gets wetter.
    assert soft[0] < soft[1] < soft[2]


# ── Weather Monte Carlo ───────────────────────────────────────────────────

def test_weather_mc_exact_at_dry_nominal_when_no_rain(monza_surface):
    surf, buckets, track, veh, loader = monza_surface
    veh.fuel_mass = loader.race_info().fuel_load_kg
    rs = RaceSimulator(LapSimulator(track, veh), weather=WeatherModel.constant(0.0))
    comps = loader.tyre_compounds()
    P = loader.race_info().pit_lane_delta_s
    s = rs.simulate(20, RaceStrategy("M-S", comps["Medium"],
                                     [PitStop(10, comps["Soft"], P)]), step_size=120.0)
    # A forecast that never rains ⇒ every sampled race is the dry nominal,
    # so the distribution collapses onto the deterministic total.
    fc = WeatherForecast(rain_probability=0.0, onset_lap_mean=10, onset_lap_std=2,
                         peak_wetness_mean=0.6, peak_wetness_std=0.1, ramp_laps=2,
                         duration_laps_mean=5, duration_laps_std=1, race_laps=20)
    dist = WeatherMonteCarlo(fc, surf, WeatherModel.constant(0.0), buckets=buckets,
                             num_samples=200, seed=3).evaluate([s])
    assert dist[0].std == pytest.approx(0.0, abs=1e-6)
    assert dist[0].p50 == pytest.approx(s.total_time, abs=1e-3)


def test_weather_mc_win_probabilities_sum_to_one(monza_surface):
    surf, buckets, track, veh, loader = monza_surface
    veh.fuel_mass = loader.race_info().fuel_load_kg
    rs = RaceSimulator(LapSimulator(track, veh), weather=WeatherModel.constant(0.0))
    comps = loader.tyre_compounds()
    P = loader.race_info().pit_lane_delta_s
    a = rs.simulate(20, RaceStrategy("A", comps["Medium"], [PitStop(10, comps["Soft"], P)]), step_size=120.0)
    b = rs.simulate(20, RaceStrategy("B", comps["Soft"], [PitStop(10, comps["Medium"], P)]), step_size=120.0)
    fc = WeatherForecast(rain_probability=0.6, onset_lap_mean=8, onset_lap_std=2,
                         peak_wetness_mean=0.6, peak_wetness_std=0.15, ramp_laps=2,
                         duration_laps_mean=6, duration_laps_std=2, race_laps=20)
    dist = WeatherMonteCarlo(fc, surf, WeatherModel.constant(0.0), buckets=buckets,
                             num_samples=500, seed=4).evaluate([a, b])
    total = sum(d.win_probability for d in dist)
    assert abs(total - 1.0) < 1e-6
    for d in dist:
        assert d.p5 <= d.p50 <= d.p95
