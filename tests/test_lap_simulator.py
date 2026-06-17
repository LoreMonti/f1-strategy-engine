# =========================================================
# Tests — src/simulation/lap_simulator.py
# =========================================================

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.track import Track, TrackSegment
from src.models.vehicle import Vehicle
from src.models.tyre import SOFT, MEDIUM, HARD
from src.simulation.lap_simulator import LapSimulator


def _make_track() -> Track:
    return Track("Test Track", [
        TrackSegment("Straight", 800.0, 0.0),
        TrackSegment(
            "Corner", 200.0, 1 / 80,
            banking_deg=2.0, traction_factor=0.92, braking_severity=1.2,
        ),
    ])


def _make_vehicle() -> Vehicle:
    return Vehicle(
        name="Test Car",
        mass=798.0, fuel_mass=100.0, fuel_consumption_per_km=2.13,
        max_power=735_000.0, drag_coefficient=0.9, lift_coefficient=3.0,
        front_aero_balance=0.45, frontal_area=1.5,
        max_brake_accel=5.5 * 9.81, tyre_mu=1.7, max_speed=95.0,
        gear_ratios=None, final_drive=3.5, wheel_radius=0.33,
        max_rpm=12_000.0, idle_rpm=4_000.0, peak_torque=750.0,
        drivetrain_efficiency=0.92,
    )


class TestTrackDiscretisation(unittest.TestCase):

    def setUp(self):
        self.sim = LapSimulator(_make_track(), _make_vehicle())

    def test_points_cover_full_length(self):
        pts = self.sim.build_track_points(step_size=5.0)
        self.assertAlmostEqual(
            pts[-1]["distance"], self.sim.track.total_length, delta=1.0
        )

    def test_distances_monotonically_increasing(self):
        pts = self.sim.build_track_points(step_size=5.0)
        for p1, p2 in zip(pts, pts[1:]):
            self.assertGreaterEqual(p2["distance"], p1["distance"])

    def test_braking_pass_never_increases_limits(self):
        raw    = self.sim.build_track_points(step_size=5.0)
        braked = self.sim.compute_speed_limits_with_braking(step_size=5.0)
        for r, b in zip(raw, braked):
            self.assertLessEqual(b["speed_limit"], r["speed_limit"] + 1e-6)

    def test_finer_step_gives_more_points(self):
        coarse = self.sim.build_track_points(step_size=20.0)
        fine   = self.sim.build_track_points(step_size=5.0)
        self.assertGreater(len(fine), len(coarse))


class TestSingleLap(unittest.TestCase):

    def setUp(self):
        self.sim = LapSimulator(_make_track(), _make_vehicle())

    def test_lap_time_positive(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        self.assertGreater(result["total_time"], 0.0)

    def test_tyre_wear_increases(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        self.assertGreater(result["final_tyre_wear"], 0.0)

    def test_tyre_wear_below_one(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        self.assertLessEqual(result["final_tyre_wear"], 1.0)

    def test_fuel_decreases(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        self.assertLess(result["final_fuel_mass"], self.sim.vehicle.fuel_mass)

    def test_soft_faster_than_hard_fresh(self):
        soft = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        hard = self.sim.simulate(step_size=10.0, tyre_compound=HARD)
        self.assertLess(soft["total_time"], hard["total_time"])

    def test_soft_wears_faster_than_hard(self):
        soft = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        hard = self.sim.simulate(step_size=10.0, tyre_compound=HARD)
        self.assertGreater(soft["final_tyre_wear"], hard["final_tyre_wear"])

    def test_lap_time_in_plausible_range(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=MEDIUM)
        self.assertGreater(result["total_time"], 15.0)
        self.assertLess(result["total_time"], 90.0)

    def test_result_has_required_keys(self):
        result = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        for key in ("total_time", "points", "final_tyre_wear", "final_fuel_mass"):
            self.assertIn(key, result)


class TestMultiLap(unittest.TestCase):

    def setUp(self):
        self.sim = LapSimulator(_make_track(), _make_vehicle())

    def test_correct_lap_count(self):
        result = self.sim.simulate_multiple_laps(num_laps=3, step_size=10.0)
        self.assertEqual(len(result["laps"]), 3)

    def test_wear_increases_each_lap(self):
        result = self.sim.simulate_multiple_laps(num_laps=4, step_size=10.0)
        wears  = [lap["final_tyre_wear"] for lap in result["laps"]]
        for w1, w2 in zip(wears, wears[1:]):
            self.assertGreater(w2, w1)

    def test_fuel_decreases_each_lap(self):
        result = self.sim.simulate_multiple_laps(num_laps=3, step_size=10.0)
        fuels  = [lap["final_fuel_mass"] for lap in result["laps"]]
        for f1, f2 in zip(fuels, fuels[1:]):
            self.assertLess(f2, f1)

    def test_lap_times_degrade_over_stint(self):
        # Isolate tyre degradation from fuel burn, which can mask early-stint
        # degradation on this very short synthetic circuit.
        self.sim.vehicle.fuel_consumption_per_km = 0.0
        result = self.sim.simulate_multiple_laps(num_laps=8, step_size=10.0)
        t3 = result["laps"][2]["lap_time"]
        t8 = result["laps"][7]["lap_time"]
        self.assertGreaterEqual(t8, t3)

    def test_invalid_num_laps_raises(self):
        with self.assertRaises(ValueError):
            self.sim.simulate_multiple_laps(num_laps=0, step_size=10.0)


class TestTelemetry(unittest.TestCase):

    def setUp(self):
        self.sim = LapSimulator(_make_track(), _make_vehicle())

    def test_has_expected_channels(self):
        result    = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        telemetry = self.sim.build_telemetry(result["points"])
        for key in ("s", "v_kmh", "a", "gear", "rpm", "tyre_wear", "fuel_mass"):
            self.assertIn(key, telemetry)

    def test_speed_always_positive(self):
        result    = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        telemetry = self.sim.build_telemetry(result["points"])
        self.assertTrue((telemetry["v_kmh"] > 0.0).all())

    def test_distance_array_matches_points(self):
        result    = self.sim.simulate(step_size=10.0, tyre_compound=SOFT)
        telemetry = self.sim.build_telemetry(result["points"])
        self.assertEqual(len(telemetry["s"]), len(result["points"]))


if __name__ == "__main__":
    unittest.main()
