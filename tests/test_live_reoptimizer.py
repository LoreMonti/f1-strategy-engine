"""Tests for the live race re-optimiser (Safety-Car pit decisions)."""
import pytest

from main import build_vehicle, build_track_from_yaml
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.optimization.live_reoptimizer import LiveReoptimizer, RaceState
from src.models.strategy import RaceStrategy, PitStop


@pytest.fixture(scope="module")
def setup():
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    ri = loader.race_info()
    veh = build_vehicle(loader.vehicle_setup() or {})
    veh.fuel_mass = ri.fuel_load_kg
    veh.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    rs = RaceSimulator(LapSimulator(track, veh))
    comps = loader.tyre_compounds()
    N = 16
    race = rs.simulate(N, RaceStrategy("M-S", comps["Medium"],
                       [PitStop(9, comps["Soft"], ri.pit_lane_delta_s)]), step_size=50.0)
    reopt = LiveReoptimizer(rs.lap_sim, N, ri.pit_lane_delta_s,
                            list(comps.values()), min_stint_laps=3, step_size=50.0)
    return race, reopt, N


def test_racestate_reconstruction(setup):
    race, _, _ = setup
    st = RaceState.from_result(race, 5)
    assert st.lap == 5
    assert st.compound.name == "Medium"     # before the L9 pit
    assert st.tyre_age == 5                  # 5 laps on the set
    assert st.used_compounds == {"Medium"}


def test_options_sorted_and_flags_consistent(setup):
    race, reopt, _ = setup
    st = RaceState.from_result(race, 4)
    opts = reopt.decide(st, regime="green", max_remaining_stops=1)
    assert len(opts) > 0
    times = [o.remaining_time for o in opts]
    assert times == sorted(times)            # ranked best-first
    assert opts[0].delta_vs_best == 0.0


def test_reoptimiser_applies_per_lap_weather(setup):
    # Regression: the re-optimiser must drive the lap simulator with the real
    # per-lap track wetness, not a stale value — otherwise it is weather-blind
    # (and would fit slicks on a still-wet track). Here a constant-0.5 weather
    # must leave the lap simulator at wetness 0.5 after simulating the remainder.
    from src.models.weather import WeatherModel
    race, reopt0, N = setup
    wet = WeatherModel.constant(0.5)
    reopt = LiveReoptimizer(reopt0.lap_sim, N, 24.5, reopt0.compounds,
                            min_stint_laps=3, step_size=50.0, weather=wet)
    state = RaceState.from_result(race, 5)
    reopt.lap_sim.track_wetness = 0.0           # stale value
    reopt._sim_remaining(state, [], set(), 1.0)
    assert abs(reopt.lap_sim.track_wetness - 0.5) < 1e-9


def test_safety_car_pulls_the_stop_no_later_than_green(setup):
    race, reopt, _ = setup
    st = RaceState.from_result(race, 4)
    g = reopt.decide(st, regime="green", max_remaining_stops=1)
    s = reopt.decide(st, regime="sc", sc_duration=4, max_remaining_stops=1)

    def first_pit_lap(opt):
        return opt.remaining_pits[0][0] if opt.remaining_pits else 999

    # The Safety Car never makes you pit LATER than the green-flag optimum;
    # it pulls the stop into the discounted window (earlier or equal).
    assert first_pit_lap(s[0]) <= first_pit_lap(g[0])
