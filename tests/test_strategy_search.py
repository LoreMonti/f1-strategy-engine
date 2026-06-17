# =========================================================
# Tests — optimization/strategy_search.py + race_simulator
# =========================================================

import sys
import os
import unittest


from src.models.track import Track, TrackSegment
from src.models.vehicle import Vehicle
from src.models.tyre import SOFT, MEDIUM, HARD
from src.models.strategy import RaceStrategy, PitStop, LapResult, RaceResult
from src.simulation.lap_simulator import LapSimulator
from src.simulation.race_simulator import RaceSimulator
from src.optimization.strategy_search import (
    generate_candidate_strategies,
    generate_and_simulate,
)
from src.optimization.strategy_optimizer import DPStrategyOptimizer


def _make_simulators():
    track = Track("Test Track", [
        TrackSegment("Straight", 800.0, 0.0),
        TrackSegment("Corner",   200.0, 1 / 80),
    ])
    vehicle = Vehicle(
        name="Test Car",
        mass=798.0, fuel_mass=100.0, fuel_consumption_per_km=2.13,
        max_power=735_000.0, drag_coefficient=0.9, lift_coefficient=3.0,
        front_aero_balance=0.45, frontal_area=1.5,
        max_brake_accel=5.5 * 9.81, tyre_mu=1.7, max_speed=95.0,
        gear_ratios=None, final_drive=3.5, wheel_radius=0.33,
        max_rpm=12_000.0, idle_rpm=4_000.0, peak_torque=750.0,
        drivetrain_efficiency=0.92,
    )
    lap_sim  = LapSimulator(track, vehicle)
    race_sim = RaceSimulator(lap_sim)
    return lap_sim, race_sim


class TestStrategyGeneration(unittest.TestCase):

    def test_generates_strategies(self):
        strategies = generate_candidate_strategies(
            num_laps=6, compounds=[SOFT, MEDIUM],
            pit_loss=22.0, min_stint_laps=2, max_stops=1,
            require_two_compounds=True,
        )
        self.assertGreater(len(strategies), 0)

    def test_no_consecutive_same_compound(self):
        strategies = generate_candidate_strategies(
            num_laps=6, compounds=[SOFT, MEDIUM, HARD],
            pit_loss=22.0, min_stint_laps=2, max_stops=2,
            require_two_compounds=True,
        )
        for s in strategies:
            stints = s.build_stints(6)
            for st1, st2 in zip(stints, stints[1:]):
                self.assertNotEqual(st1.compound.name, st2.compound.name)

    def test_two_compound_rule_enforced(self):
        strategies = generate_candidate_strategies(
            num_laps=6, compounds=[SOFT, MEDIUM],
            pit_loss=22.0, min_stint_laps=2, max_stops=2,
            require_two_compounds=True,
        )
        for s in strategies:
            stints    = s.build_stints(6)
            compounds = {st.compound.name for st in stints}
            self.assertGreaterEqual(len(compounds), 2)

    def test_min_stint_laps_respected(self):
        min_laps   = 3
        strategies = generate_candidate_strategies(
            num_laps=8, compounds=[SOFT, MEDIUM],
            pit_loss=22.0, min_stint_laps=min_laps, max_stops=1,
            require_two_compounds=True,
        )
        for s in strategies:
            for st in s.build_stints(8):
                self.assertGreaterEqual(st.length, min_laps)

    def test_max_stops_one_gives_one_stop_only(self):
        strategies = generate_candidate_strategies(
            num_laps=6, compounds=[SOFT, MEDIUM, HARD],
            pit_loss=22.0, min_stint_laps=2, max_stops=1,
            require_two_compounds=True,
        )
        for s in strategies:
            self.assertEqual(len(s.pit_stops), 1)

    def test_two_stop_strategies_present_when_allowed(self):
        strategies = generate_candidate_strategies(
            num_laps=8, compounds=[SOFT, MEDIUM, HARD],
            pit_loss=22.0, min_stint_laps=2, max_stops=2,
            require_two_compounds=True,
        )
        two_stops = [s for s in strategies if len(s.pit_stops) == 2]
        self.assertGreater(len(two_stops), 0)


class TestRaceSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lap_sim, cls.race_sim = _make_simulators()

    def _one_stop(self):
        return RaceStrategy(
            "Soft-Medium L4", SOFT,
            [PitStop(lap=4, new_compound=MEDIUM, time_loss=22.0)],
        )

    def test_result_is_typed(self):
        result = self.race_sim.simulate(6, self._one_stop(), step_size=10.0)
        self.assertIsInstance(result, RaceResult)
        for lr in result.laps:
            self.assertIsInstance(lr, LapResult)

    def test_correct_lap_count(self):
        result = self.race_sim.simulate(5, RaceStrategy("Soft", SOFT, []), step_size=10.0)
        self.assertEqual(result.num_laps, 5)
        self.assertEqual(len(result.laps), 5)

    def test_pit_stop_registered(self):
        result = self.race_sim.simulate(6, self._one_stop(), step_size=10.0)
        self.assertEqual(result.num_stops, 1)

    def test_pit_lap_has_correct_time_loss(self):
        pit_loss = 22.0
        result   = self.race_sim.simulate(6, self._one_stop(), step_size=10.0)
        pit_laps = [lr for lr in result.laps if lr.pit_stop]
        self.assertEqual(len(pit_laps), 1)
        self.assertAlmostEqual(pit_laps[0].pit_time_loss, pit_loss)

    def test_compound_changes_after_pit(self):
        result = self.race_sim.simulate(
            6,
            RaceStrategy("Soft-Hard L3", SOFT, [PitStop(3, HARD, 22.0)]),
            step_size=10.0,
        )
        self.assertTrue(all(lr.compound == "Soft" for lr in result.laps if lr.lap < 3))
        self.assertTrue(all(lr.compound == "Hard" for lr in result.laps if lr.lap >= 3))

    def test_cumulative_time_monotonically_increasing(self):
        result = self.race_sim.simulate(5, RaceStrategy("Soft", SOFT, []), step_size=10.0)
        times  = [lr.cumulative_time for lr in result.laps]
        for t1, t2 in zip(times, times[1:]):
            self.assertGreater(t2, t1)

    def test_total_time_equals_sum_of_lap_times(self):
        result   = self.race_sim.simulate(6, self._one_stop(), step_size=10.0)
        sum_laps = sum(lr.lap_time for lr in result.laps)
        self.assertAlmostEqual(result.total_time, sum_laps, places=6)

    def test_invalid_num_laps_raises(self):
        with self.assertRaises(ValueError):
            self.race_sim.simulate(0, RaceStrategy("Soft", SOFT, []), step_size=10.0)

    def test_results_sorted_fastest_first(self):
        results = generate_and_simulate(
            race_simulator=self.race_sim,
            num_laps=6, compounds=[SOFT, MEDIUM],
            pit_loss=22.0, min_stint_laps=2, max_stops=1,
            require_two_compounds=True, step_size=10.0,
        )
        times = [r.total_time for r in results]
        for t1, t2 in zip(times, times[1:]):
            self.assertLessEqual(t1, t2)

    def test_higher_pit_loss_never_improves_time(self):
        def _best(pit_loss):
            return generate_and_simulate(
                race_simulator=self.race_sim,
                num_laps=6, compounds=[SOFT, MEDIUM],
                pit_loss=pit_loss, min_stint_laps=2, max_stops=1,
                require_two_compounds=True, step_size=10.0,
            )[0].total_time

        self.assertGreaterEqual(_best(30.0), _best(20.0))


class TestDPStrategyOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lap_sim, cls.race_sim = _make_simulators()

    def test_stint_table_uses_fresh_tyres_at_late_start(self):
        optimizer = DPStrategyOptimizer(
            race_simulator=self.race_sim,
            min_stint_laps=2,
            verbose=False,
        )
        optimizer._build_stint_table(
            num_laps=6,
            compounds=[MEDIUM],
            step_size=20.0,
        )

        start_lap = 4
        length = 2
        dp_time = optimizer._get_stint_time(MEDIUM, start_lap, length)

        fuel_burn_per_lap = (
            self.lap_sim.vehicle.fuel_consumption_per_km
            * self.lap_sim.track.total_length
            / 1000.0
        )
        fuel_mass = self.lap_sim.vehicle.fuel_mass - fuel_burn_per_lap * (start_lap - 1)
        speed = 1.0
        gear = 1
        tyre_wear = 0.0
        tyre_temp = MEDIUM.pit_temperature
        manual_time = 0.0

        for lap_idx in range(length):
            result = self.lap_sim.simulate(
                step_size=20.0,
                tyre_compound=MEDIUM,
                initial_speed=speed,
                initial_gear=gear,
                initial_tyre_wear=tyre_wear,
                initial_tyre_temperature=tyre_temp,
                initial_fuel_mass=fuel_mass,
            )
            # The stint table includes the empirical degradation overlay
            # (deg_s_per_lap × laps_on_tyre, 0 on the fresh lap), so the manual
            # reference must add it too — this is the DP↔RaceSimulator coherence
            # guaranteed by the step-1 fix.
            manual_time += result["total_time"] + MEDIUM.deg_s_per_lap * lap_idx
            speed = result["final_speed"]
            gear = result["final_gear"]
            tyre_wear = result["final_tyre_wear"]
            tyre_temp = result["final_tyre_temperature"]
            fuel_mass = result["final_fuel_mass"]

        self.assertIsNotNone(dp_time)
        self.assertAlmostEqual(dp_time, manual_time, places=6)


if __name__ == "__main__":
    unittest.main()
