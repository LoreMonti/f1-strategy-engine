# =========================================================
# Lap Simulator
#
# Responsibilities:
#   1. Discretise the track into spatial sample points.
#   2. Compute braking-constrained speed limits (backward pass).
#   3. Run the forward integration lap-by-lap using Integrator.
#   4. Build telemetry arrays from raw simulation points.
#
# Race-level logic (pit stops, strategy) lives in RaceSimulator.
# Plotting lives in visualization/.
# =========================================================

from __future__ import annotations

import numpy as np

from src.models.track import Track
from src.models.vehicle import Vehicle
from src.models.tyre import TyreCompound, TyreState, MEDIUM
from src.simulation.integrator import Integrator, StepResult
from src.utils.units import m_to_km


class LapSimulator:
    """
    Simulates one or more consecutive laps on a fixed track.

    Parameters
    ----------
    track : Track
    vehicle : Vehicle
    """

    def __init__(self, track: Track, vehicle: Vehicle,
                 track_wetness: float = 0.0) -> None:
        self.track      = track
        self.vehicle    = vehicle
        self.integrator = Integrator(vehicle)
        # Level A static weather: track wetness [0 = dry, 1 = soaked].
        # Scales every tyre's grip via TyreCompound.wet_grip_factor().
        self.track_wetness = max(0.0, min(1.0, track_wetness))
        # Cache keyed by (step_size, grip_multiplier) — reused across
        # all simulate() calls with the same step, saving ~95% of the
        # compute_speed_limits_with_braking() overhead in strategy search.
        self._track_points_cache: dict[tuple[float, float], list[dict]] = {}

    def __repr__(self) -> str:
        return (
            f"LapSimulator(track='{self.track.name}', "
            f"vehicle='{self.vehicle.name}')"
        )

    # ------------------------------------------------------------------ #
    # Track discretisation                                                 #
    # ------------------------------------------------------------------ #

    def _segment_speed_limits(self, grip_multiplier: float = 1.0) -> dict[str, float]:
        """Maximum speed per segment name [m/s]."""
        return {
            seg.name: min(
                self.integrator._max_corner_speed(
                    seg.curvature,
                    seg.banking_deg,
                    grip_multiplier,
                    fuel_mass=None,
                ),
                self.vehicle.max_speed,
            )
            for seg in self.track.segments
        }

    def _classify_corner_phase(self, curvature: float, progress: float) -> str:
        """Return 'straight' | 'entry' | 'mid' | 'exit'."""
        if curvature <= 0.0:
            return "straight"
        if progress < 1.0 / 3.0:
            return "entry"
        if progress < 2.0 / 3.0:
            return "mid"
        return "exit"

    def _make_track_point(
        self,
        segment,
        distance: float,
        progress: float,
        speed_limits: dict[str, float],
    ) -> dict:
        """Build a single track-point dict."""
        return {
            "distance":        distance,
            "segment":         segment.name,
            "curvature":       segment.curvature,
            "grip_factor":     segment.grip_factor,
            "banking_deg":     segment.banking_deg,
            "braking_severity": segment.braking_severity,
            "traction_factor": segment.traction_factor,
            "speed_limit":     speed_limits[segment.name],
            "corner_phase":    self._classify_corner_phase(segment.curvature, progress),
        }

    def build_track_points(
        self,
        step_size: float = 5.0,
        grip_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Discretise the track into spatial sample points.

        Segment boundaries are never duplicated: when the end of one
        segment coincides with the start of the next, the boundary
        point is updated in-place so the new segment owns it.
        """
        speed_limits = self._segment_speed_limits(grip_multiplier)
        points: list[dict] = []
        distance = 0.0

        for segment in self.track.segments:
            seg_start = distance
            seg_end   = seg_start + segment.length

            # --- Segment start point ---
            start_pt = self._make_track_point(segment, seg_start, 0.0, speed_limits)
            if points and abs(points[-1]["distance"] - seg_start) < 1e-9:
                points[-1] = start_pt
            else:
                points.append(start_pt)

            # --- Interior points ---
            pos = seg_start + step_size
            while pos < seg_end:
                progress = (pos - seg_start) / max(segment.length, 1e-6)
                points.append(
                    self._make_track_point(segment, pos, progress, speed_limits)
                )
                pos += step_size

            # --- Segment end point ---
            points.append(
                self._make_track_point(segment, seg_end, 1.0, speed_limits)
            )

            distance = seg_end

        return points

    def compute_speed_limits_with_braking(
        self,
        step_size: float = 5.0,
        grip_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Apply a backward braking pass to the raw speed-limit profile.

        Ensures that the car can always brake to the next corner limit
        within the available deceleration budget.
        """
        points = self.build_track_points(step_size, grip_multiplier)
        limits = [p["speed_limit"] for p in points]

        eff_brake = self.vehicle.max_brake_accel * grip_multiplier

        for i in range(len(limits) - 2, -1, -1):
            ds = points[i + 1]["distance"] - points[i]["distance"]
            if ds <= 0.0:
                continue
            max_before = (limits[i + 1] ** 2 + 2.0 * eff_brake * ds) ** 0.5
            limits[i] = min(limits[i], max_before)

        for point, lim in zip(points, limits):
            point["speed_limit"] = lim

        return points

    # ------------------------------------------------------------------ #
    # Single-lap simulation                                                #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        step_size: float = 5.0,
        tyre_compound: TyreCompound = MEDIUM,
        initial_speed: float = 1.0,
        initial_gear: int = 1,
        initial_tyre_wear: float = 0.0,
        initial_tyre_temperature: float = 70.0,
        initial_fuel_mass: float | None = None,
    ) -> dict:
        """
        Simulate one lap and return a raw result dict.

        The result dict contains:
        - ``total_time``       : lap time [s]
        - ``points``           : list of per-step state dicts
        - ``final_*``          : final state values
        - ``tyre_compound``    : compound name
        """
        # Backward braking pass is keyed on (step_size, compound base_grip).
        # Using grip=1.0 for all compounds produces wrong compound ordering at
        # coarse step sizes: Hard (low grip → low eff_brake) cannot brake to the
        # optimistic limits set for grip=1.0, so it passes corners faster than
        # physically possible.  Using the compound's base_grip gives the correct
        # conservative speed profile for each compound.
        # Fold the wet grip factor into the braking-pass grip estimate so the
        # speed-limit cache reflects the reduced wet grip (otherwise braking
        # would be computed against dry corner limits).
        approx_grip = tyre_compound.base_grip * tyre_compound.wet_grip_factor(self.track_wetness)
        _cache_key = (step_size, round(approx_grip, 3))
        if _cache_key not in self._track_points_cache:
            self._track_points_cache[_cache_key] = (
                self.compute_speed_limits_with_braking(step_size, grip_multiplier=approx_grip)
            )
        points = self._track_points_cache[_cache_key]

        fuel_mass = (
            initial_fuel_mass
            if initial_fuel_mass is not None
            else self.vehicle.fuel_mass
        )

        front_tyre = TyreState(
            compound=tyre_compound,
            wear=initial_tyre_wear,
            temperature=initial_tyre_temperature,
            track_wetness=self.track_wetness,
        )
        rear_tyre = TyreState(
            compound=tyre_compound,
            wear=initial_tyre_wear,
            temperature=initial_tyre_temperature,
            track_wetness=self.track_wetness,
        )

        speed        = initial_speed
        current_gear = initial_gear
        total_time   = 0.0

        # Initial state point
        results = [self._initial_point(points[0], speed, current_gear, fuel_mass, front_tyre, rear_tyre)]

        for i in range(len(points) - 1):
            ds = points[i + 1]["distance"] - points[i]["distance"]
            if ds <= 0.0:
                continue

            step = self.integrator.step(
                ds=ds,
                speed=speed,
                gear=current_gear,
                fuel_mass=fuel_mass,
                front_tyre=front_tyre,
                rear_tyre=rear_tyre,
                current_point=points[i],
                next_point=points[i + 1],
                previous_speed=speed,
            )

            total_time   += step.dt
            speed         = step.speed
            current_gear  = step.gear
            fuel_mass     = step.fuel_mass

            results.append(self._step_to_point(points[i + 1], step, total_time, tyre_compound))

        last = results[-1]
        return {
            "track":                   self.track.name,
            "vehicle":                 self.vehicle.name,
            "tyre_compound":           tyre_compound.name,
            "total_time":              total_time,
            "points":                  results,
            "final_speed":             last["speed"],
            "final_gear":              last["gear"],
            "final_tyre_wear":         last["tyre_wear"],
            "final_tyre_temperature":  last["tyre_temperature"],
            "final_front_tyre_wear":   last["front_tyre_wear"],
            "final_rear_tyre_wear":    last["rear_tyre_wear"],
            "final_front_tyre_temperature": last["front_tyre_temperature"],
            "final_rear_tyre_temperature":  last["rear_tyre_temperature"],
            "final_fuel_mass":         last["fuel_mass"],
            "final_vehicle_mass":      last["vehicle_mass"],
        }

    # ------------------------------------------------------------------ #
    # Multi-lap simulation                                                 #
    # ------------------------------------------------------------------ #

    def simulate_multiple_laps(
        self,
        num_laps: int,
        step_size: float = 5.0,
        tyre_compound: TyreCompound = MEDIUM,
    ) -> dict:
        """
        Simulate consecutive laps, carrying state from one lap to the next.

        Returns a dict with ``laps`` (list of per-lap dicts) and aggregates.
        """
        if num_laps < 1:
            raise ValueError("num_laps must be >= 1.")

        lap_results  = []
        total_time   = 0.0
        speed        = 1.0
        gear         = 1
        tyre_wear    = 0.0
        tyre_temp    = tyre_compound.pit_temperature
        fuel_mass    = self.vehicle.fuel_mass

        for lap_n in range(1, num_laps + 1):
            result = self.simulate(
                step_size=step_size,
                tyre_compound=tyre_compound,
                initial_speed=speed,
                initial_gear=gear,
                initial_tyre_wear=tyre_wear,
                initial_tyre_temperature=tyre_temp,
                initial_fuel_mass=fuel_mass,
            )

            total_time += result["total_time"]

            pts            = result["points"]
            initial_fuel   = pts[0]["fuel_mass"]
            final_fuel     = result["final_fuel_mass"]
            max_speed      = max(p["speed"] for p in pts)
            final_grip     = pts[-1]["grip_multiplier"]
            prev_time      = lap_results[-1]["lap_time"] if lap_results else None
            delta          = result["total_time"] - prev_time if prev_time is not None else 0.0

            lap_results.append({
                "lap":                    lap_n,
                "lap_time":               result["total_time"],
                "delta_lap_time":         delta,
                "cumulative_time":        total_time,
                "final_speed":            result["final_speed"],
                "final_gear":             result["final_gear"],
                "final_tyre_wear":        result["final_tyre_wear"],
                "final_tyre_temperature": result["final_tyre_temperature"],
                "final_grip_multiplier":  final_grip,
                "initial_fuel_mass":      initial_fuel,
                "final_fuel_mass":        final_fuel,
                "fuel_used":              initial_fuel - final_fuel,
                "final_vehicle_mass":     result["final_vehicle_mass"],
                "max_speed":              max_speed,
                "points":                 pts,
            })

            # Carry state forward
            speed     = result["final_speed"]
            gear      = result["final_gear"]
            tyre_wear = result["final_tyre_wear"]
            tyre_temp = result["final_tyre_temperature"]
            fuel_mass = result["final_fuel_mass"]

        return {
            "track":         self.track.name,
            "vehicle":       self.vehicle.name,
            "tyre_compound": tyre_compound.name,
            "num_laps":      num_laps,
            "total_time":    total_time,
            "laps":          lap_results,
        }

    # ------------------------------------------------------------------ #
    # Telemetry builder                                                    #
    # ------------------------------------------------------------------ #

    def build_telemetry(self, simulation_points: list[dict]) -> dict:
        """
        Convert raw simulation points into numpy arrays for plotting.

        Extracts all channels stored per-step and computes derived
        signals (throttle %, brake %, engine power, grip usage, etc.).
        """
        # Extract raw channels
        keys_direct = [
            "distance", "speed", "segment", "gear", "rpm",
            "tyre_wear", "tyre_temperature", "grip_multiplier",
            "front_tyre_wear", "rear_tyre_wear",
            "front_tyre_temperature", "rear_tyre_temperature",
            "front_grip_multiplier", "rear_grip_multiplier",
            "fuel_mass", "vehicle_mass",
            "front_workload", "rear_workload",
            "axle_workload_imbalance", "dynamic_front_aero_balance",
            "acceleration",
        ]

        raw: dict[str, list] = {k: [] for k in keys_direct}

        for pt in simulation_points:
            for k in keys_direct:
                raw[k].append(pt.get(k, 0.0))

        s       = np.array(raw["distance"])
        v       = np.array(raw["speed"])
        v_kmh   = v * 3.6
        a       = np.array(raw["acceleration"])
        gear_arr = np.array(raw["gear"], dtype=int)
        rpm_arr  = np.array(raw["rpm"])

        # Throttle / brake as percentage
        max_accel = max(float(np.max(a)),  1e-6)
        max_brake = max(float(-np.min(a)), 1e-6)
        throttle  = np.clip(a / max_accel,  0.0, 1.0) * 100.0
        brake     = np.clip(-a / max_brake, 0.0, 1.0) * 100.0

        # Aero / force arrays
        drag          = np.array([self.vehicle.drag_force(vi)  for vi in v])
        downforce_arr = np.array([self.vehicle.downforce(vi)   for vi in v])
        gm_arr        = np.array(raw["grip_multiplier"])
        engine_force  = np.array([
            self.vehicle.engine_force(vi, gi, gmi)
            for vi, gi, gmi in zip(v, gear_arr, gm_arr)
        ])
        engine_power_kw = engine_force * v / 1000.0

        max_grip_force = np.array([
            self.vehicle.max_grip_force(vi, gmi)
            for vi, gmi in zip(v, gm_arr)
        ])
        required_force = np.abs(a) * self.vehicle.mass
        grip_usage = np.clip(
            required_force / np.maximum(max_grip_force, 1e-6), 0.0, 1.0
        ) * 100.0

        fuel_arr = np.array(raw["fuel_mass"])

        return {
            "s":                          s,
            "v":                          v,
            "v_kmh":                      v_kmh,
            "a":                          a,
            "throttle":                   throttle,
            "brake":                      brake,
            "drag":                       drag,
            "downforce":                  downforce_arr,
            "engine_force":               engine_force,
            "engine_power_kw":            engine_power_kw,
            "max_grip_force":             max_grip_force,
            "grip_usage":                 grip_usage,
            "gear":                       gear_arr,
            "rpm":                        rpm_arr,
            "engine_torque":              np.array([self.vehicle.engine_torque(r) for r in rpm_arr]),
            "tyre_wear":                  np.array(raw["tyre_wear"]),
            "tyre_temperature":           np.array(raw["tyre_temperature"]),
            "grip_multiplier":            gm_arr,
            "front_tyre_wear":            np.array(raw["front_tyre_wear"]),
            "rear_tyre_wear":             np.array(raw["rear_tyre_wear"]),
            "front_tyre_temperature":     np.array(raw["front_tyre_temperature"]),
            "rear_tyre_temperature":      np.array(raw["rear_tyre_temperature"]),
            "front_grip_multiplier":      np.array(raw["front_grip_multiplier"]),
            "rear_grip_multiplier":       np.array(raw["rear_grip_multiplier"]),
            "fuel_mass":                  fuel_arr,
            "vehicle_mass":               np.array(raw["vehicle_mass"]),
            "fuel_used":                  self.vehicle.fuel_mass - fuel_arr,
            "front_downforce":            np.array([
                self.vehicle.front_downforce_dynamic(vi) for vi in v
            ]),
            "rear_downforce":             np.array([
                self.vehicle.rear_downforce_dynamic(vi) for vi in v
            ]),
            "front_normal_force":         np.array([
                self.vehicle.front_normal_force_dynamic(vi) for vi in v
            ]),
            "rear_normal_force":          np.array([
                self.vehicle.rear_normal_force_dynamic(vi) for vi in v
            ]),
            "front_workload":             np.array(raw["front_workload"]),
            "rear_workload":              np.array(raw["rear_workload"]),
            "axle_workload_imbalance":    np.array(raw["axle_workload_imbalance"]),
            "dynamic_front_aero_balance": np.array(raw["dynamic_front_aero_balance"]),
            "segments":                   raw["segment"],
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _initial_point(
        self,
        track_point: dict,
        speed: float,
        gear: int,
        fuel_mass: float,
        front_tyre: TyreState,
        rear_tyre: TyreState,
    ) -> dict:
        """Build the first simulation point (before any integration step)."""
        return {
            "distance":                   track_point["distance"],
            "segment":                    track_point["segment"],
            "speed":                      speed,
            "speed_limit":                track_point["speed_limit"],
            "acceleration":               0.0,
            "time":                       0.0,
            "gear":                       gear,
            "rpm":                        self.vehicle.rpm_for_gear(speed, gear),
            "tyre_wear":                  0.5 * (front_tyre.wear + rear_tyre.wear),
            "tyre_temperature":           0.5 * (front_tyre.temperature + rear_tyre.temperature),
            "grip_multiplier":            0.5 * (front_tyre.grip_multiplier() + rear_tyre.grip_multiplier()),
            "front_tyre_wear":            front_tyre.wear,
            "rear_tyre_wear":             rear_tyre.wear,
            "front_tyre_temperature":     front_tyre.temperature,
            "rear_tyre_temperature":      rear_tyre.temperature,
            "front_grip_multiplier":      front_tyre.grip_multiplier(),
            "rear_grip_multiplier":       rear_tyre.grip_multiplier(),
            "tyre_compound":              front_tyre.compound.name,
            "fuel_mass":                  fuel_mass,
            "vehicle_mass":               self.vehicle.current_mass(fuel_mass),
            "dynamic_front_aero_balance": self.vehicle.front_aero_balance,
            "front_workload":             0.0,
            "rear_workload":              0.0,
            "axle_workload_imbalance":    0.0,
            "lateral_force":              0.0,
        }

    @staticmethod
    def _step_to_point(
        track_point: dict,
        step: StepResult,
        elapsed_time: float,
        tyre_compound: TyreCompound,
    ) -> dict:
        """Convert a StepResult + track metadata into a simulation-point dict."""
        return {
            "distance":                   track_point["distance"],
            "segment":                    track_point["segment"],
            "speed":                      step.speed,
            "speed_limit":                step.speed_limit,
            "acceleration":               step.acceleration,
            "time":                       elapsed_time,
            "gear":                       step.gear,
            "rpm":                        step.rpm,
            "tyre_wear":                  step.tyre_wear,
            "tyre_temperature":           step.tyre_temperature,
            "grip_multiplier":            step.grip_multiplier,
            "front_tyre_wear":            step.front_tyre_wear,
            "rear_tyre_wear":             step.rear_tyre_wear,
            "front_tyre_temperature":     step.front_tyre_temperature,
            "rear_tyre_temperature":      step.rear_tyre_temperature,
            "front_grip_multiplier":      step.front_grip_multiplier,
            "rear_grip_multiplier":       step.rear_grip_multiplier,
            "tyre_compound":              tyre_compound.name,
            "fuel_mass":                  step.fuel_mass,
            "vehicle_mass":               step.vehicle_mass,
            "front_workload":             step.front_workload,
            "rear_workload":              step.rear_workload,
            "axle_workload_imbalance":    step.axle_workload_imbalance,
            "dynamic_front_aero_balance": step.dynamic_front_aero_balance,
            "lateral_force":              step.lateral_force,
        }
    
