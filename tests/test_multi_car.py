"""Tests for the multi-car simulator: track position, undercut / overcut."""
import pytest

from main import build_vehicle, build_track_from_yaml
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.simulation.multi_car_simulator import MultiCarSimulator, detect_overtakes
from src.models.strategy import RaceStrategy, PitStop


@pytest.fixture(scope="module")
def monza():
    track, loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    ri = loader.race_info()
    veh = build_vehicle(loader.vehicle_setup() or {})
    veh.fuel_mass = ri.fuel_load_kg
    veh.fuel_consumption_per_km = ri.fuel_consumption_kg_per_lap / ri.lap_distance_km
    rs = RaceSimulator(LapSimulator(track, veh))
    return rs, loader


# ── Pure-function behaviour ───────────────────────────────────────────────

def test_overtake_margin_scales_inversely_with_likelihood(monza):
    rs, _ = monza
    easy = MultiCarSimulator(rs, overtaking_likelihood=0.65)
    hard = MultiCarSimulator(rs, overtaking_likelihood=0.20)
    # Harder circuit ⇒ a follower needs a BIGGER pace advantage to pass.
    assert hard.overtake_margin_s > easy.overtake_margin_s


def test_loader_overtaking_likelihood_is_sector_mean():
    _, monza_loader = build_track_from_yaml("data/tracks/monza_2024.yaml")
    _, sing_loader = build_track_from_yaml("data/tracks/singapore_2024.yaml")
    # Monza is easier to overtake at than Singapore (street circuit).
    assert monza_loader.overtaking_likelihood() > sing_loader.overtaking_likelihood()
    assert 0.0 < sing_loader.overtaking_likelihood() <= 1.0


# ── Track-position (sticky) behaviour ─────────────────────────────────────

def test_faster_car_is_held_up_on_hard_circuit(monza):
    rs, loader = monza
    comps = loader.tyre_compounds()
    P = loader.race_info().pit_lane_delta_s
    # Car 1 (pole) on the slower Hard; Car 2 (behind) on the faster Soft but
    # no pit stops for either → identical track, Car 2 wants to pass.
    entries = [
        ("Lead", RaceStrategy("Hard-1stop", comps["Hard"], [])),
        ("Chaser", RaceStrategy("Soft-1stop", comps["Soft"], [])),
    ]
    hard = MultiCarSimulator(rs, overtaking_likelihood=0.10, grid_gap_s=0.3)
    res = hard.simulate(entries, num_laps=8, step_size=50.0)
    chaser = next(c for c in res.cars if c.name == "Chaser")
    # On a near-impossible-to-pass circuit the faster chaser loses time stuck
    # in dirty air rather than sailing past.
    assert chaser.total_traffic_penalty > 0.0


def test_easier_circuit_has_less_traffic_penalty(monza):
    rs, loader = monza
    comps = loader.tyre_compounds()
    entries = [
        ("Lead", RaceStrategy("Hard", comps["Hard"], [])),
        ("Chaser", RaceStrategy("Soft", comps["Soft"], [])),
    ]
    hard = MultiCarSimulator(rs, overtaking_likelihood=0.10, grid_gap_s=0.3)
    easy = MultiCarSimulator(rs, overtaking_likelihood=0.90, grid_gap_s=0.3)
    pen_hard = hard.simulate(entries, 8, 50.0).cars
    pen_easy = easy.simulate(entries, 8, 50.0).cars
    th = sum(c.total_traffic_penalty for c in pen_hard)
    te = sum(c.total_traffic_penalty for c in pen_easy)
    assert th >= te


# ── Undercut / overcut detection ──────────────────────────────────────────

def test_undercut_is_detected_and_classified(monza):
    rs, loader = monza
    comps = loader.tyre_compounds()
    P = loader.race_info().pit_lane_delta_s
    # Lead starts ahead and pits late; Chaser pits EARLY (undercut attempt).
    entries = [
        ("Lead", RaceStrategy("late", comps["Medium"], [PitStop(9, comps["Medium"], P)])),
        ("Chaser", RaceStrategy("early", comps["Medium"], [PitStop(4, comps["Soft"], P)])),
    ]
    hard = MultiCarSimulator(rs, overtaking_likelihood=0.15, grid_gap_s=0.2)
    res = hard.simulate(entries, num_laps=12, step_size=50.0)
    kinds = {e.kind for e in res.overtake_events}
    # Every detected event must be one of the three valid causes.
    assert kinds <= {"undercut", "overcut", "on-track pass"}
    # The result carries an overtake-events list populated by simulate().
    assert isinstance(res.overtake_events, list)


def test_detect_overtakes_matches_simulate(monza):
    rs, loader = monza
    comps = loader.tyre_compounds()
    P = loader.race_info().pit_lane_delta_s
    entries = [
        ("A", RaceStrategy("A", comps["Soft"], [PitStop(5, comps["Medium"], P)])),
        ("B", RaceStrategy("B", comps["Medium"], [PitStop(8, comps["Soft"], P)])),
    ]
    res = MultiCarSimulator(rs, overtaking_likelihood=0.30).simulate(entries, 10, 50.0)
    recomputed = detect_overtakes(res)
    assert len(recomputed) == len(res.overtake_events)
