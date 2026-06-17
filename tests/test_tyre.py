# =========================================================
# Tests — src/models/tyre.py
# =========================================================

import sys
import os
import unittest


from src.models.tyre import (
    TyreState, SOFT, MEDIUM, HARD, INTERMEDIATE, WET, TYRE_COMPOUNDS,
)


class TestWetGripCrossover(unittest.TestCase):
    """The wet/slick crossover that the weather model relies on."""

    def _best(self, wetness):
        # Effective grip = base_grip × wet_grip_factor; highest wins.
        cands = {
            "slick": SOFT.base_grip * SOFT.wet_grip_factor(wetness),
            "inter": INTERMEDIATE.base_grip * INTERMEDIATE.wet_grip_factor(wetness),
            "wet":   WET.base_grip * WET.wet_grip_factor(wetness),
        }
        return max(cands, key=cands.get)

    def test_slick_best_when_dry(self):
        self.assertEqual(self._best(0.0), "slick")

    def test_intermediate_best_when_damp(self):
        self.assertEqual(self._best(0.5), "inter")

    def test_wet_best_when_soaked(self):
        self.assertEqual(self._best(0.9), "wet")

    def test_slick_grip_collapses_in_the_wet(self):
        self.assertLess(SOFT.wet_grip_factor(1.0), 0.6)


class TestTyreCompound(unittest.TestCase):

    def test_wear_increment_positive(self):
        self.assertGreater(SOFT.wear_increment(1000.0), 0.0)

    def test_wear_increment_scales_with_distance(self):
        w1 = SOFT.wear_increment(100.0)
        w2 = SOFT.wear_increment(200.0)
        self.assertAlmostEqual(w2, 2 * w1)

    def test_hot_tyre_wears_more(self):
        cool = SOFT.wear_increment(1000.0, temperature=80.0)
        hot  = SOFT.wear_increment(1000.0, temperature=130.0)
        self.assertGreater(hot, cool)

    def test_cold_tyre_wears_more_than_nominal(self):
        nominal = SOFT.wear_increment(1000.0, temperature=SOFT.optimal_temperature)
        cold    = SOFT.wear_increment(1000.0, temperature=50.0)
        self.assertGreater(cold, nominal)

    def test_wear_grip_monotonically_decreasing(self):
        grips = [SOFT.wear_grip_multiplier(w) for w in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]]
        for g1, g2 in zip(grips, grips[1:]):
            self.assertGreaterEqual(g1, g2)

    def test_wear_grip_never_below_min(self):
        for wear in [0.0, 0.5, 0.9, 1.0, 1.5]:
            self.assertGreaterEqual(
                SOFT.wear_grip_multiplier(wear),
                SOFT.min_grip_multiplier,
            )

    def test_temperature_grip_peaks_at_optimal(self):
        optimal = SOFT.temperature_grip_multiplier(SOFT.optimal_temperature)
        cold    = SOFT.temperature_grip_multiplier(SOFT.cold_temperature - 10)
        hot     = SOFT.temperature_grip_multiplier(SOFT.overheating_temperature + 10)
        self.assertGreater(optimal, cold)
        self.assertGreater(optimal, hot)

    def test_combined_grip_never_below_min(self):
        for wear in [0.0, 0.5, 1.0]:
            for temp in [40.0, 95.0, 140.0]:
                self.assertGreaterEqual(
                    SOFT.grip_multiplier(wear, temp),
                    SOFT.min_grip_multiplier,
                )

    def test_hard_wears_slower_than_soft(self):
        self.assertLess(
            HARD.wear_increment(1000.0),
            SOFT.wear_increment(1000.0),
        )

    def test_grip_hierarchy(self):
        self.assertGreater(SOFT.base_grip, MEDIUM.base_grip)
        self.assertGreater(MEDIUM.base_grip, HARD.base_grip)

    def test_compounds_dict_complete(self):
        # Slicks plus the wet-weather compounds (Level A/B weather model).
        self.assertEqual(
            set(TYRE_COMPOUNDS.keys()),
            {"soft", "medium", "hard", "intermediate", "wet"},
        )


class TestTyreState(unittest.TestCase):

    def test_wear_increases_after_update(self):
        state = TyreState(compound=SOFT)
        initial = state.wear
        state.update(distance_m=500.0, speed=60.0, acceleration=5.0, grip_usage=0.8)
        self.assertGreater(state.wear, initial)

    def test_wear_can_exceed_design_limit(self):
        state = TyreState(compound=SOFT, wear=0.999)
        for _ in range(10):
            state.update(distance_m=1000.0, speed=50.0, acceleration=5.0, grip_usage=1.0)
        self.assertGreater(state.wear, 1.0)
        self.assertGreaterEqual(state.grip_multiplier(), SOFT.min_grip_multiplier)

    def test_temperature_changes_after_update(self):
        state = TyreState(compound=SOFT, temperature=70.0)
        state.update(distance_m=500.0, speed=60.0, acceleration=5.0, grip_usage=0.9)
        self.assertNotEqual(state.temperature, 70.0)

    def test_temperature_stays_within_limits(self):
        state = TyreState(compound=SOFT, temperature=70.0)
        for _ in range(20):
            state.update(distance_m=200.0, speed=80.0, acceleration=8.0, grip_usage=1.0)
        self.assertGreaterEqual(state.temperature, state.min_temperature)
        self.assertLessEqual(state.temperature, state.max_temperature)

    def test_reset_clears_wear_and_distance(self):
        state = TyreState(compound=SOFT, wear=0.4, distance_on_tyre=5000.0)
        state.reset()
        self.assertEqual(state.wear, 0.0)
        self.assertEqual(state.distance_on_tyre, 0.0)

    def test_reset_changes_compound(self):
        state = TyreState(compound=SOFT)
        state.reset(compound=MEDIUM)
        self.assertEqual(state.compound.name, "Medium")

    def test_reset_sets_pit_temperature(self):
        state = TyreState(compound=SOFT)
        state.reset(compound=MEDIUM)
        self.assertEqual(state.temperature, MEDIUM.pit_temperature)

    def test_warmup_grip_increases_with_distance(self):
        state = TyreState(compound=SOFT)
        g0 = state._warmup_multiplier()
        state.distance_on_tyre = 3000.0
        g1 = state._warmup_multiplier()
        self.assertGreater(g1, g0)

    def test_grip_decreases_with_wear(self):
        fresh = TyreState(compound=SOFT, distance_on_tyre=5000.0)
        worn  = TyreState(compound=SOFT, wear=0.6, distance_on_tyre=5000.0)
        self.assertGreater(fresh.grip_multiplier(), worn.grip_multiplier())


if __name__ == "__main__":
    unittest.main()
