# =========================================================
# Race Simulator
#
# Executes a RaceStrategy lap by lap, handling pit stops,
# compound changes, and fuel/tyre state continuity.
#
# Returns typed RaceResult / LapResult objects instead of
# plain dicts.
# =========================================================

from __future__ import annotations

from src.models.strategy import RaceStrategy, LapResult, RaceResult
from src.simulation.lap_simulator import LapSimulator


class RaceSimulator:
    """
    Simulates a full race given a strategy.

    Parameters
    ----------
    lap_simulator : LapSimulator
        Pre-built lap simulator (track + vehicle already bound).
    """

    def __init__(self, lap_simulator: LapSimulator, weather=None) -> None:
        self.lap_sim = lap_simulator
        # Optional WeatherModel (Level B). When set, the track wetness is
        # updated lap-by-lap before each lap is simulated, so a drying or
        # worsening track is reflected during the race. When None, the lap
        # simulator's static track_wetness (Level A) is used unchanged.
        self.weather = weather

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        num_laps: int,
        strategy: RaceStrategy,
        step_size: float = 5.0,
    ) -> RaceResult:
        """
        Simulate a race using the given strategy.

        Pit stops:
        - add a fixed time loss on the lap they occur,
        - reset tyre wear and temperature to the new compound's pit value,
        - preserve fuel mass and carry-over speed/gear.

        Parameters
        ----------
        num_laps : int
            Total race distance in laps.
        strategy : RaceStrategy
            Compound sequence and pit stop schedule.
        step_size : float
            Spatial integration step [m].

        Returns
        -------
        RaceResult
            Fully typed race result.
        """
        if num_laps < 1:
            raise ValueError("num_laps must be >= 1.")

        lap_results: list[LapResult] = []
        total_time = 0.0

        compound       = strategy.initial_compound
        speed          = 1.0
        gear           = 1
        tyre_wear      = 0.0
        tyre_temp      = compound.pit_temperature
        fuel_mass      = self.lap_sim.vehicle.fuel_mass
        laps_on_tyre   = 0

        for lap_n in range(1, num_laps + 1):

            # --- Dynamic weather (Level B): set the track wetness for this
            #     lap before simulating it. ------------------------------
            if self.weather is not None:
                self.lap_sim.track_wetness = self.weather.wetness(lap_n)

            # --- Pit stop processing (before this lap) ------------------
            pit = strategy.pit_stop_on_lap(lap_n)
            pit_loss    = 0.0
            pit_flag    = False

            if pit is not None:
                pit_loss       = pit.time_loss
                compound       = pit.new_compound
                tyre_wear      = 0.0
                tyre_temp      = compound.pit_temperature
                pit_flag       = True
                laps_on_tyre   = 0

            laps_on_tyre += 1

            # --- Lap simulation -----------------------------------------
            raw = self.lap_sim.simulate(
                step_size=step_size,
                tyre_compound=compound,
                initial_speed=speed,
                initial_gear=gear,
                initial_tyre_wear=tyre_wear,
                initial_tyre_temperature=tyre_temp,
                initial_fuel_mass=fuel_mass,
            )

            # Empirical degradation overlay: compensates for thermal/chemical
            # tyre performance loss that the physics grip model understimates.
            # Zero on lap 1 of a stint (fresh tyre), increasing linearly.
            deg_penalty  = compound.deg_s_per_lap * (laps_on_tyre - 1)
            raw_lap_time = raw["total_time"] + deg_penalty
            lap_time     = raw_lap_time + pit_loss
            total_time += lap_time

            pts           = raw["points"]
            initial_fuel  = pts[0]["fuel_mass"]
            final_fuel    = raw["final_fuel_mass"]
            max_speed     = max(p["speed"] for p in pts)
            final_grip    = pts[-1]["grip_multiplier"]
            prev_time     = lap_results[-1].lap_time if lap_results else None
            delta         = lap_time - prev_time if prev_time is not None else 0.0

            lap_result = LapResult(
                lap=lap_n,
                compound=compound.name,
                raw_lap_time=raw_lap_time,
                pit_time_loss=pit_loss,
                lap_time=lap_time,
                delta_lap_time=delta,
                cumulative_time=total_time,
                pit_stop=pit_flag,
                final_tyre_wear=raw["final_tyre_wear"],
                final_tyre_temperature=raw["final_tyre_temperature"],
                final_grip_multiplier=final_grip,
                final_front_tyre_wear=raw["final_front_tyre_wear"],
                final_rear_tyre_wear=raw["final_rear_tyre_wear"],
                final_front_tyre_temperature=raw["final_front_tyre_temperature"],
                final_rear_tyre_temperature=raw["final_rear_tyre_temperature"],
                initial_fuel_mass=initial_fuel,
                final_fuel_mass=final_fuel,
                fuel_used=initial_fuel - final_fuel,
                final_vehicle_mass=raw["final_vehicle_mass"],
                max_speed=max_speed,
                final_speed=raw["final_speed"],
                final_gear=raw["final_gear"],
            )
            lap_results.append(lap_result)

            # Carry state to next lap
            speed     = raw["final_speed"]
            gear      = raw["final_gear"]
            tyre_wear = raw["final_tyre_wear"]
            tyre_temp = raw["final_tyre_temperature"]
            fuel_mass = raw["final_fuel_mass"]

        # --- Aggregates -------------------------------------------------
        fastest  = min(lap_results, key=lambda lr: lr.raw_lap_time)
        avg_raw  = sum(lr.raw_lap_time for lr in lap_results) / len(lap_results)
        pit_loss_total = sum(lr.pit_time_loss for lr in lap_results)

        return RaceResult(
            track=self.lap_sim.track.name,
            vehicle=self.lap_sim.vehicle.name,
            strategy=strategy.name,
            num_laps=num_laps,
            total_time=total_time,
            total_pit_loss=pit_loss_total,
            average_raw_lap_time=avg_raw,
            laps=lap_results,
            fastest_lap=fastest,
            stints=strategy.build_stints(num_laps),
        )

    def __repr__(self) -> str:
        return f"RaceSimulator(lap_simulator={self.lap_sim!r})"
