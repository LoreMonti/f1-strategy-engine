# =========================================================
# Tests — src/models/vehicle.py
# =========================================================

import sys
import os
import unittest


from src.models.vehicle import Vehicle


def _make_vehicle(**kwargs) -> Vehicle:
    defaults = dict(
        name="Test Car",
        mass=798.0,
        fuel_mass=100.0,
        fuel_consumption_per_km=2.13,
        max_power=735_000.0,
        drag_coefficient=0.9,
        lift_coefficient=3.0,
        front_aero_balance=0.45,
        frontal_area=1.5,
        max_brake_accel=5.5 * 9.81,
        tyre_mu=1.7,
        max_speed=95.0,
        gear_ratios=None,
        final_drive=3.5,
        wheel_radius=0.33,
        max_rpm=12_000.0,
        idle_rpm=4_000.0,
        peak_torque=750.0,
        drivetrain_efficiency=0.92,
    )
    defaults.update(kwargs)
    return Vehicle(**defaults)


class TestVehicleValidation(unittest.TestCase):

    def test_negative_mass_raises(self):
        with self.assertRaises(ValueError):
            _make_vehicle(mass=-100.0)

    def test_zero_wheel_radius_raises(self):
        with self.assertRaises(ValueError):
            _make_vehicle(wheel_radius=0.0)

    def test_drivetrain_efficiency_above_one_raises(self):
        with self.assertRaises(ValueError):
            _make_vehicle(drivetrain_efficiency=1.5)

    def test_zero_front_aero_balance_raises(self):
        with self.assertRaises(ValueError):
            _make_vehicle(front_aero_balance=0.0)


class TestVehicleMass(unittest.TestCase):

    def test_current_mass_includes_fuel(self):
        self.assertAlmostEqual(_make_vehicle().current_mass(), 898.0)

    def test_current_mass_with_custom_fuel(self):
        self.assertAlmostEqual(_make_vehicle().current_mass(fuel_mass=50.0), 848.0)

    def test_current_mass_zero_fuel(self):
        self.assertAlmostEqual(_make_vehicle().current_mass(fuel_mass=0.0), 798.0)


class TestAerodynamics(unittest.TestCase):

    def test_drag_zero_at_rest(self):
        self.assertEqual(_make_vehicle().drag_force(0.0), 0.0)

    def test_drag_increases_with_speed(self):
        v = _make_vehicle()
        self.assertGreater(v.drag_force(50.0), v.drag_force(30.0))

    def test_downforce_zero_at_rest(self):
        self.assertEqual(_make_vehicle().downforce(0.0), 0.0)

    def test_downforce_increases_with_speed(self):
        v = _make_vehicle()
        self.assertGreater(v.downforce(80.0), v.downforce(40.0))

    def test_front_plus_rear_equals_total(self):
        v     = _make_vehicle()
        speed = 60.0
        self.assertAlmostEqual(
            v.front_downforce_dynamic(speed) + v.rear_downforce_dynamic(speed),
            v.downforce(speed),
            places=5,
        )

    def test_dynamic_balance_within_limits(self):
        v = _make_vehicle()
        for speed in [10.0, 50.0, 90.0]:
            bal = v.dynamic_front_aero_balance(speed)
            self.assertGreaterEqual(bal, v.min_front_aero_balance)
            self.assertLessEqual(bal, v.max_front_aero_balance)


class TestEngine(unittest.TestCase):

    def test_torque_positive_at_idle(self):
        v = _make_vehicle()
        self.assertGreater(v.engine_torque(v.idle_rpm), 0.0)

    def test_torque_zero_above_rev_limiter(self):
        v = _make_vehicle()
        self.assertEqual(v.engine_torque(v.max_rpm + 1000.0), 0.0)

    def test_rpm_increases_with_speed(self):
        v = _make_vehicle()
        self.assertGreater(v.rpm_for_gear(50.0, 3), v.rpm_for_gear(30.0, 3))

    def test_select_gear_stays_below_max_rpm(self):
        v = _make_vehicle()
        for speed in [10.0, 30.0, 60.0, 90.0]:
            _, rpm = v.select_gear(speed)
            self.assertLessEqual(rpm, v.max_rpm + 1.0)

    def test_engine_force_positive_at_low_speed(self):
        self.assertGreater(_make_vehicle().engine_force(10.0), 0.0)

    def test_acceleration_positive_at_low_speed(self):
        self.assertGreater(_make_vehicle().acceleration(10.0), 0.0)

    def test_acceleration_reduces_with_worn_tyres(self):
        v = _make_vehicle()
        self.assertGreaterEqual(
            v.acceleration(50.0, grip_multiplier=1.0),
            v.acceleration(50.0, grip_multiplier=0.7),
        )


class TestGripForce(unittest.TestCase):

    def test_grip_increases_with_speed(self):
        v = _make_vehicle()
        self.assertGreater(v.max_grip_force(80.0), v.max_grip_force(20.0))

    def test_grip_decreases_with_degradation(self):
        v = _make_vehicle()
        self.assertGreater(
            v.max_grip_force(60.0, grip_multiplier=1.0),
            v.max_grip_force(60.0, grip_multiplier=0.8),
        )


if __name__ == "__main__":
    unittest.main()
