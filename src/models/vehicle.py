# =========================================================
# Vehicle model
#
# Parametric F1-style vehicle with:
# - aerodynamics (drag, downforce, dynamic balance)
# - engine (torque curve, gear selection)
# - tyre grip budget
# - fuel mass tracking
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np

from src.utils.constants import AIR_DENSITY, GRAVITY


# Static weight distribution (front / rear)
_STATIC_FRONT_BIAS: float = 0.46
_STATIC_REAR_BIAS: float  = 0.54


@dataclass
class Vehicle:
    """
    F1-style vehicle specification.

    Parameters
    ----------
    name : str
        Descriptive label.
    mass : float
        Dry chassis mass [kg] (without fuel).
    fuel_mass : float
        Initial fuel load [kg].
    fuel_consumption_per_km : float
        Fuel burn rate [kg/km].
    max_power : float
        Peak engine power [W].
    drag_coefficient : float
        Aerodynamic drag coefficient (Cd).
    lift_coefficient : float
        Aerodynamic downforce coefficient (Cl, positive = downforce).
    frontal_area : float
        Vehicle frontal area [m²].
    max_brake_accel : float
        Maximum braking deceleration [m/s²].
    tyre_mu : float
        Nominal tyre friction coefficient.
    max_speed : float
        Absolute speed cap [m/s].
    gear_ratios : list[float] | None
        Gear ratios from 1st to top. Defaults to an 8-speed set.
    final_drive : float
        Final drive ratio.
    wheel_radius : float
        Driven wheel radius [m].
    max_rpm : float
        Rev limiter [RPM].
    idle_rpm : float
        Minimum engine RPM.
    peak_torque : float
        Peak engine torque [Nm].
    drivetrain_efficiency : float
        Transmission efficiency [0–1].
    front_aero_balance : float
        Static front aero balance (fraction of downforce on front axle).
    """

    name: str
    mass: float
    fuel_mass: float
    fuel_consumption_per_km: float
    max_power: float
    drag_coefficient: float
    lift_coefficient: float
    frontal_area: float
    max_brake_accel: float
    tyre_mu: float
    max_speed: float
    gear_ratios: list[float] | None
    final_drive: float
    wheel_radius: float
    max_rpm: float
    idle_rpm: float
    peak_torque: float
    drivetrain_efficiency: float

    # ERS / MGU-K deployment power [kW].
    # 0 = ERS not modelled (backward-compatible default).
    # Race mode typical Monza: ~80 kW (net after harvesting).
    # Qualifying mode: 120 kW (full deployment, no energy saving).
    ers_power_kw: float = 0.0

    front_aero_balance: float = 0.45

    # Dynamic aero balance limits
    min_front_aero_balance: float = 0.40
    max_front_aero_balance: float = 0.52

    # Aero migration factors under load
    braking_aero_migration: float     = 0.035
    acceleration_aero_migration: float = 0.025
    speed_aero_migration: float       = 0.020

    aero_reference_speed: float = 80.0  # [m/s]

    # Set in __post_init__
    upshift_rpm: float = field(init=False, repr=False)
    downshift_rpm: float = field(init=False, repr=False)
    max_gear_number: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gear_ratios is None:
            self.gear_ratios = [3.20, 2.45, 1.95, 1.60, 1.35, 1.15, 0.98, 0.85]

        self.upshift_rpm   = 11_500.0
        self.downshift_rpm = 8_500.0
        self.max_gear_number = len(self.gear_ratios)

        # Basic sanity checks
        if self.mass <= 0:
            raise ValueError("Vehicle mass must be positive.")
        if self.wheel_radius <= 0:
            raise ValueError("Wheel radius must be positive.")
        if not (0.0 < self.drivetrain_efficiency <= 1.0):
            raise ValueError("Drivetrain efficiency must be in (0, 1].")
        if not (0.0 < self.front_aero_balance < 1.0):
            raise ValueError("front_aero_balance must be in (0, 1).")

    # ------------------------------------------------------------------ #
    # Mass                                                                 #
    # ------------------------------------------------------------------ #

    def current_mass(self, fuel_mass: float | None = None) -> float:
        """Total vehicle mass including fuel [kg]."""
        return self.mass + (fuel_mass if fuel_mass is not None else self.fuel_mass)

    # ------------------------------------------------------------------ #
    # Aerodynamics                                                         #
    # ------------------------------------------------------------------ #

    def _aero_pressure(self, speed: float) -> float:
        """Dynamic pressure term 0.5 * ρ * v² [Pa]."""
        return 0.5 * AIR_DENSITY * speed ** 2

    def drag_force(self, speed: float) -> float:
        """Aerodynamic drag force [N]."""
        return self._aero_pressure(speed) * self.drag_coefficient * self.frontal_area

    def downforce(self, speed: float) -> float:
        """Total aerodynamic downforce [N]."""
        return self._aero_pressure(speed) * self.lift_coefficient * self.frontal_area

    def dynamic_front_aero_balance(
        self,
        speed: float,
        acceleration: float = 0.0,
    ) -> float:
        """
        Estimate dynamic front aero balance fraction.

        Braking shifts the aero platform forward; acceleration and
        high speed shift it rearward.
        """
        speed_factor = min(max(speed / self.aero_reference_speed, 0.0), 1.5)

        braking_intensity     = min(max(-acceleration / self.max_brake_accel, 0.0), 1.0)
        acceleration_intensity = min(max(acceleration / 20.0, 0.0), 1.0)

        balance = self.front_aero_balance
        balance += self.braking_aero_migration * braking_intensity * speed_factor
        balance -= self.acceleration_aero_migration * acceleration_intensity * speed_factor
        balance -= self.speed_aero_migration * min(speed / self.max_speed, 1.0)

        return float(np.clip(balance, self.min_front_aero_balance, self.max_front_aero_balance))

    def front_downforce_dynamic(self, speed: float, acceleration: float = 0.0) -> float:
        return self.downforce(speed) * self.dynamic_front_aero_balance(speed, acceleration)

    def rear_downforce_dynamic(self, speed: float, acceleration: float = 0.0) -> float:
        return self.downforce(speed) * (1.0 - self.dynamic_front_aero_balance(speed, acceleration))

    # ------------------------------------------------------------------ #
    # Normal forces                                                        #
    # ------------------------------------------------------------------ #

    def normal_force(self, speed: float, fuel_mass: float | None = None) -> float:
        """Total vertical load on tyres [N]."""
        return self.current_mass(fuel_mass) * GRAVITY + self.downforce(speed)

    def front_normal_force_dynamic(
        self,
        speed: float,
        acceleration: float = 0.0,
        fuel_mass: float | None = None,
    ) -> float:
        static = _STATIC_FRONT_BIAS * self.current_mass(fuel_mass) * GRAVITY
        return static + self.front_downforce_dynamic(speed, acceleration)

    def rear_normal_force_dynamic(
        self,
        speed: float,
        acceleration: float = 0.0,
        fuel_mass: float | None = None,
    ) -> float:
        static = _STATIC_REAR_BIAS * self.current_mass(fuel_mass) * GRAVITY
        return static + self.rear_downforce_dynamic(speed, acceleration)

    # ------------------------------------------------------------------ #
    # Grip                                                                 #
    # ------------------------------------------------------------------ #

    def max_grip_force(
        self,
        speed: float,
        grip_multiplier: float = 1.0,
        fuel_mass: float | None = None,
    ) -> float:
        """Maximum tyre grip force [N]."""
        return self.tyre_mu * grip_multiplier * self.normal_force(speed, fuel_mass)

    # ------------------------------------------------------------------ #
    # Powertrain                                                           #
    # ------------------------------------------------------------------ #

    def engine_torque(self, rpm: float) -> float:
        """
        Simplified F1-style torque curve.

        - Linear rise from idle to 7 000 RPM
        - Flat peak from 7 000 to 10 500 RPM
        - Linear fall to rev limiter
        """
        if rpm < self.idle_rpm:
            return 0.4 * self.peak_torque

        if rpm < 7_000.0:
            return self.peak_torque * (
                0.65 + 0.35 * (rpm - self.idle_rpm) / (7_000.0 - self.idle_rpm)
            )

        if rpm < 10_500.0:
            return self.peak_torque

        if rpm <= self.max_rpm:
            return self.peak_torque * (
                1.0 - 0.15 * (rpm - 10_500.0) / (self.max_rpm - 10_500.0)
            )

        return 0.0

    def rpm_for_gear(self, speed: float, gear: int) -> float:
        """Engine RPM at a given speed and gear."""
        if speed <= 0.1:
            return self.idle_rpm

        wheel_omega  = speed / self.wheel_radius
        engine_omega = wheel_omega * self.gear_ratios[gear - 1] * self.final_drive
        rpm = engine_omega * 60.0 / (2.0 * math.pi)
        return max(self.idle_rpm, rpm)

    def select_gear(self, speed: float) -> tuple[int, float]:
        """
        Select the highest gear that keeps RPM below the rev limiter.

        Returns (gear, rpm).
        """
        valid = [
            (g, self.rpm_for_gear(speed, g))
            for g in range(1, self.max_gear_number + 1)
            if self.rpm_for_gear(speed, g) <= self.max_rpm
        ]

        if valid:
            return max(valid, key=lambda x: x[1])

        top = self.max_gear_number
        return top, self.rpm_for_gear(speed, top)

    def update_gear(self, speed: float, current_gear: int) -> int:
        """Shift up/down based on RPM thresholds (with hysteresis)."""
        current_gear = int(np.clip(current_gear, 1, self.max_gear_number))
        rpm = self.rpm_for_gear(speed, current_gear)

        if rpm >= self.upshift_rpm and current_gear < self.max_gear_number:
            return current_gear + 1
        if rpm <= self.downshift_rpm and current_gear > 1:
            return current_gear - 1
        return current_gear

    def engine_force(
        self,
        speed: float,
        gear: int | None = None,
        grip_multiplier: float = 1.0,
        fuel_mass: float | None = None,
        ers_active: bool = False,
    ) -> float:
        """
        Wheel drive force, limited by engine torque, power, and tyre grip.

        Parameters
        ----------
        ers_active : bool
            If True, add MGU-K deployment power (``ers_power_kw``) on top of
            the ICE ``max_power``.  The integrator sets this flag whenever the
            car is on a straight segment (curvature == 0).
        """
        speed = max(speed, 1.0)

        if gear is None:
            gear, rpm = self.select_gear(speed)
        else:
            rpm = self.rpm_for_gear(speed, gear)

        wheel_torque = (
            self.engine_torque(rpm)
            * self.gear_ratios[gear - 1]
            * self.final_drive
            * self.drivetrain_efficiency
        )

        wheel_force = wheel_torque / self.wheel_radius

        # ICE power + optional ERS MGU-K boost on straights
        ers_w = self.ers_power_kw * 1_000.0 if ers_active else 0.0
        power_limited_force = (self.max_power + ers_w) / speed

        grip_limited_force  = self.max_grip_force(speed, grip_multiplier, fuel_mass)

        return min(wheel_force, power_limited_force, grip_limited_force)

    def acceleration(
        self,
        speed: float,
        gear: int | None = None,
        grip_multiplier: float = 1.0,
        fuel_mass: float | None = None,
        ers_active: bool = False,
    ) -> float:
        """
        Net longitudinal acceleration [m/s²].

        Limited by engine force, tyre grip, and drag.
        """
        traction_force = min(
            self.engine_force(speed, gear, grip_multiplier, fuel_mass, ers_active),
            self.max_grip_force(speed, grip_multiplier, fuel_mass),
        )
        net_force    = traction_force - self.drag_force(speed)
        current_mass = self.current_mass(fuel_mass)

        # Small rotational inertia penalty
        mass_factor = self.mass / current_mass
        return (net_force / current_mass) * (0.92 + 0.08 * mass_factor)

    def __repr__(self) -> str:
        return (
            f"Vehicle(name='{self.name}', "
            f"mass={self.mass:.0f} kg, "
            f"max_power={self.max_power / 1000:.0f} kW)"
        )
