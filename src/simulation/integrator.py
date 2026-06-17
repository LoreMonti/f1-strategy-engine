# =========================================================
# Physics integrator
#
# Advances the vehicle state over a single spatial step [ds].
#
# This module knows nothing about laps, races, or strategies.
# It only answers one question:
#   "Given the current state and the next track point,
#    what is the new state after ds metres?"
#
# Used by LapSimulator, which calls step() for every spatial
# sample along the track.
# =========================================================

from __future__ import annotations
import math
from dataclasses import dataclass

from src.models.vehicle import Vehicle
from src.models.tyre import TyreState
from src.utils.constants import GRAVITY


# ------------------------------------------------------------------ #
# State containers                                                    #
# ------------------------------------------------------------------ #

@dataclass
class VehicleState:
    """
    Instantaneous vehicle state at a single track point.

    All values are SI units unless noted.
    """
    distance: float         # position along the lap [m]
    speed: float            # longitudinal speed [m/s]
    gear: int
    rpm: float
    fuel_mass: float        # [kg]
    elapsed_time: float     # time from lap start [s]


@dataclass
class StepResult:
    """
    Output of a single integrator step.

    Contains the new VehicleState plus derived quantities needed by
    the simulator and telemetry builder.
    """
    # Kinematics
    speed: float
    speed_limit: float
    acceleration: float
    dt: float               # time elapsed in this step [s]

    # Powertrain
    gear: int
    rpm: float

    # Fuel
    fuel_mass: float
    vehicle_mass: float

    # Tyre (averaged front/rear)
    tyre_wear: float
    tyre_temperature: float
    grip_multiplier: float

    # Per-axle
    front_tyre_wear: float
    rear_tyre_wear: float
    front_tyre_temperature: float
    rear_tyre_temperature: float
    front_grip_multiplier: float
    rear_grip_multiplier: float

    # Dynamics
    front_workload: float
    rear_workload: float
    axle_workload_imbalance: float
    dynamic_front_aero_balance: float
    lateral_force: float


# ------------------------------------------------------------------ #
# Phase tables (constant, defined once)                               #
# ------------------------------------------------------------------ #

_PHASE_TRACTION: dict[str, float] = {
    "straight": 1.00,
    "entry":    0.96,
    "mid":      0.92,
    "exit":     0.88,
}

_PHASE_WORKLOAD: dict[str, float] = {
    "straight": 1.00,
    "entry":    1.15,
    "mid":      1.05,
    "exit":     1.10,
}


# ------------------------------------------------------------------ #
# Integrator                                                          #
# ------------------------------------------------------------------ #

class Integrator:
    """
    Single-step spatial integrator for an F1-style vehicle.

    The integrator is stateless: it takes the current state and
    track-point data, and returns the result for the next point.
    All mutable state (tyre, fuel, time) is managed by the caller
    (LapSimulator).
    """

    def __init__(self, vehicle: Vehicle) -> None:
        self.vehicle = vehicle

    # ---------------------------------------------------------------- #
    # Public interface                                                   #
    # ---------------------------------------------------------------- #

    def step(
        self,
        ds: float,
        speed: float,
        gear: int,
        fuel_mass: float,
        front_tyre: TyreState,
        rear_tyre: TyreState,
        current_point: dict,
        next_point: dict,
        previous_speed: float,
    ) -> StepResult:
        """
        Advance the vehicle over a spatial step of length ds.

        Parameters
        ----------
        ds : float
            Step length [m]. Must be positive.
        speed : float
            Current speed [m/s].
        gear : int
            Current gear.
        fuel_mass : float
            Current fuel mass [kg].
        front_tyre : TyreState
            Front tyre state (mutated in-place).
        rear_tyre : TyreState
            Rear tyre state (mutated in-place).
        current_point : dict
            Track-point dict for the current position.
        next_point : dict
            Track-point dict for the next position.
        previous_speed : float
            Speed at the previous track point (for braking check) [m/s].

        Returns
        -------
        StepResult
            New state and all derived quantities.
        """
        # ---- Gear & RPM ------------------------------------------------
        gear  = self.vehicle.update_gear(speed, gear)
        rpm   = self.vehicle.rpm_for_gear(speed, gear)

        # ---- Grip budget -----------------------------------------------
        front_grip = front_tyre.grip_multiplier()
        rear_grip  = rear_tyre.grip_multiplier()
        grip_multiplier = 0.5 * (front_grip + rear_grip)

        # ---- Speed limit at next point ---------------------------------
        local_corner_limit = self._max_corner_speed(
            curvature=next_point["curvature"],
            banking_deg=next_point["banking_deg"],
            grip_multiplier=grip_multiplier,
            fuel_mass=fuel_mass,
        )
        local_speed_limit = min(
            local_corner_limit,
            self.vehicle.max_speed,
            next_point["speed_limit"],
        )

        # ---- Longitudinal dynamics -------------------------------------
        corner_phase = current_point["corner_phase"]

        effective_long_grip = (
            grip_multiplier
            * current_point["traction_factor"]
            * _PHASE_TRACTION[corner_phase]
        )

        vehicle_mass = self.vehicle.current_mass(fuel_mass)

        # ERS MGU-K is deployed on straights only (curvature == 0).
        # The integrator detects this from the corner_phase flag set by
        # LapSimulator._classify_corner_phase().
        ers_active = (corner_phase == "straight")

        desired_accel = self.vehicle.acceleration(
            speed, gear, effective_long_grip, fuel_mass, ers_active=ers_active
        )
        desired_long_force = desired_accel * vehicle_mass

        # ---- Lateral force & friction circle ---------------------------
        banking_rad = math.radians(next_point["banking_deg"])
        raw_lateral = vehicle_mass * speed ** 2 * next_point["curvature"]
        banking_assist = vehicle_mass * GRAVITY * math.sin(banking_rad)
        lateral_force = max(0.0, raw_lateral - banking_assist)

        available_grip = self.vehicle.max_grip_force(
            speed, effective_long_grip, fuel_mass
        )
        limited_long_force = self._friction_circle_limit(
            available_grip, lateral_force, desired_long_force
        )

        accel = limited_long_force / max(vehicle_mass, 1e-6)

        # ---- New speed -------------------------------------------------
        new_speed_sq = speed ** 2 + 2.0 * accel * ds
        new_speed    = max(1.0, new_speed_sq ** 0.5)

        # Braking constraint: can we actually slow to local_speed_limit?
        eff_brake = self.vehicle.max_brake_accel * grip_multiplier
        min_reachable = max(1.0, previous_speed ** 2 - 2.0 * eff_brake * ds) ** 0.5

        if local_speed_limit < previous_speed:
            speed = max(local_speed_limit, min_reachable)
        else:
            speed = min(new_speed, local_speed_limit)

        # ---- Time step -------------------------------------------------
        avg_speed = 0.5 * (previous_speed + speed)
        dt = ds / max(avg_speed, 1e-6)

        # ---- Fuel burn -------------------------------------------------
        fuel_burn_per_m = self.vehicle.fuel_consumption_per_km / 1000.0
        new_fuel = max(0.0, fuel_mass - fuel_burn_per_m * ds)

        # ---- Axle workload distribution --------------------------------
        actual_long_accel = (speed ** 2 - previous_speed ** 2) / max(2.0 * ds, 1e-6)
        long_force = abs(actual_long_accel) * vehicle_mass

        front_wl, rear_wl, dyn_aero_bal = self._axle_workload(
            corner_phase=corner_phase,
            longitudinal_force=long_force,
            lateral_force=lateral_force,
            speed=speed,
            acceleration=actual_long_accel,
            fuel_mass=new_fuel,
        )

        imbalance = abs(front_wl - rear_wl) / max(front_wl + rear_wl, 1e-6)

        # ---- Grip usage for tyre thermal model -------------------------
        max_grip = self.vehicle.max_grip_force(
            speed,
            current_point["grip_factor"] * grip_multiplier,
            new_fuel,
        )

        combined_force = (long_force ** 2 + lateral_force ** 2) ** 0.5
        combined_force *= _PHASE_WORKLOAD[corner_phase]
        combined_force *= 1.0 + 0.08 * imbalance

        grip_usage = min(combined_force / max(max_grip, 1e-6), 1.0)

        braking_sev = current_point["braking_severity"]
        thermal_usage = min(grip_usage * braking_sev, 1.0)

        front_usage = min(thermal_usage * (1.0 + imbalance), 1.0)
        rear_usage  = min(thermal_usage * (1.0 + imbalance), 1.0)

        if front_wl > rear_wl:
            front_usage = min(front_usage * 1.10, 1.0)
            rear_usage  = min(rear_usage  * 0.95, 1.0)
        elif rear_wl > front_wl:
            rear_usage  = min(rear_usage  * 1.10, 1.0)
            front_usage = min(front_usage * 0.95, 1.0)

        # ---- Update tyre states ----------------------------------------
        front_tyre.update(
            distance_m=ds,
            speed=speed,
            acceleration=actual_long_accel,
            grip_usage=front_usage,
            curvature=next_point["curvature"],
        )
        rear_tyre.update(
            distance_m=ds,
            speed=speed,
            acceleration=actual_long_accel,
            grip_usage=rear_usage,
            curvature=next_point["curvature"],
        )

        # ---- Assemble result -------------------------------------------
        return StepResult(
            speed=speed,
            speed_limit=local_speed_limit,
            # Store the ACTUAL (kinematic) longitudinal acceleration, signed:
            # positive under power, negative under braking. ``accel`` above is
            # only the powertrain/grip-limited drive force and is never negative,
            # which left the telemetry brake channel permanently at 0%.
            # This field is diagnostic only (telemetry); the integration loop
            # advances on step.speed, so this does not affect the physics.
            acceleration=actual_long_accel,
            dt=dt,
            gear=gear,
            rpm=rpm,
            fuel_mass=new_fuel,
            vehicle_mass=self.vehicle.current_mass(new_fuel),
            tyre_wear=0.5 * (front_tyre.wear + rear_tyre.wear),
            tyre_temperature=0.5 * (front_tyre.temperature + rear_tyre.temperature),
            grip_multiplier=0.5 * (front_tyre.grip_multiplier() + rear_tyre.grip_multiplier()),
            front_tyre_wear=front_tyre.wear,
            rear_tyre_wear=rear_tyre.wear,
            front_tyre_temperature=front_tyre.temperature,
            rear_tyre_temperature=rear_tyre.temperature,
            front_grip_multiplier=front_tyre.grip_multiplier(),
            rear_grip_multiplier=rear_tyre.grip_multiplier(),
            front_workload=front_wl,
            rear_workload=rear_wl,
            axle_workload_imbalance=imbalance,
            dynamic_front_aero_balance=dyn_aero_bal,
            lateral_force=lateral_force,
        )

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _max_corner_speed(
        self,
        curvature: float,
        banking_deg: float,
        grip_multiplier: float,
        fuel_mass: float | None,
        initial_guess: float = 50.0,
        tolerance: float = 1e-3,
        max_iter: int = 100,
    ) -> float:
        """Iterative maximum cornering speed (accounts for banking)."""
        if curvature <= 0.0:
            return float("inf")

        speed = initial_guess
        banking_rad  = math.radians(banking_deg)
        vehicle_mass = self.vehicle.current_mass(fuel_mass)

        for _ in range(max_iter):
            downforce = self.vehicle.downforce(speed)

            normal_load = (
                vehicle_mass * GRAVITY * math.cos(banking_rad)
                + downforce
                + vehicle_mass * speed ** 2 * curvature * math.sin(banking_rad)
            )

            avail_lateral = self.vehicle.tyre_mu * grip_multiplier * normal_load
            banking_assist = vehicle_mass * GRAVITY * math.sin(banking_rad)

            new_v_sq = (avail_lateral + banking_assist) / max(vehicle_mass * curvature, 1e-6)
            new_v    = math.sqrt(max(new_v_sq, 1.0))

            if abs(new_v - speed) < tolerance:
                return new_v
            speed = new_v

        return speed

    @staticmethod
    def _friction_circle_limit(
        available_grip: float,
        lateral_force: float,
        desired_long_force: float,
    ) -> float:
        """
        Limit longitudinal force using a simplified friction circle.

        If the lateral demand already saturates grip, no longitudinal
        force remains.
        """
        lateral_force  = max(0.0, lateral_force)
        available_grip = max(available_grip, 1e-6)

        if lateral_force >= available_grip:
            return 0.0

        remaining = (available_grip ** 2 - lateral_force ** 2) ** 0.5
        return min(abs(desired_long_force), remaining)

    def _axle_workload(
        self,
        corner_phase: str,
        longitudinal_force: float,
        lateral_force: float,
        speed: float,
        acceleration: float,
        fuel_mass: float | None,
    ) -> tuple[float, float, float]:
        """
        Estimate front/rear tyre workload split and dynamic aero balance.

        Returns (front_workload, rear_workload, dynamic_front_aero_balance).
        """
        mechanical_front = {
            "entry":    0.62,
            "mid":      0.50,
            "exit":     0.42,
            "straight": 0.50,
        }[corner_phase]

        front_load = self.vehicle.front_normal_force_dynamic(speed, acceleration, fuel_mass)
        rear_load  = self.vehicle.rear_normal_force_dynamic(speed, acceleration, fuel_mass)

        aero_front  = front_load / max(front_load + rear_load, 1e-6)
        aero_influence = min(speed / 80.0, 1.0)

        front_share = (
            (1.0 - aero_influence) * mechanical_front
            + aero_influence * aero_front
        )
        front_share = max(0.35, min(0.65, front_share))

        total_wl   = (longitudinal_force ** 2 + lateral_force ** 2) ** 0.5
        front_wl   = total_wl * front_share
        rear_wl    = total_wl * (1.0 - front_share)

        dyn_aero_bal = self.vehicle.dynamic_front_aero_balance(speed, acceleration)

        return front_wl, rear_wl, dyn_aero_bal
