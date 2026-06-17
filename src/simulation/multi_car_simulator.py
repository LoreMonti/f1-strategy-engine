# =========================================================
# Multi-Car Race Simulator
#
# Simulates N cars lap-by-lap on the same track, applying
# a simple traffic model: if a car is within `traffic_threshold_s`
# of the car directly ahead, it loses `traffic_penalty_s` per lap.
# =========================================================

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.strategy import RaceStrategy
from src.models.multi_car import CarLapResult, CarRaceResult, MultiCarRaceResult
from src.simulation.race_simulator import RaceSimulator


@dataclass
class _CarState:
    """Internal mutable state for one car during simulation."""
    name: str
    strategy: RaceStrategy
    grid_position: int
    grid_gap_s: float

    # Physics state (carried lap to lap)
    compound: object = None       # TyreCompound
    speed: float = 1.0
    gear: int = 1
    tyre_wear: float = 0.0
    tyre_temp: float = 0.0
    fuel_mass: float = 0.0
    laps_on_tyre: int = 0

    # Accumulated
    cumulative_time: float = 0.0
    total_traffic_penalty: float = 0.0
    lap_results: list = field(default_factory=list)

    def __post_init__(self):
        # cumulative_time starts with the grid gap so P2+ are already
        # "behind" P1 when the race starts
        self.cumulative_time = self.grid_gap_s


class MultiCarSimulator:
    """
    Simulates multiple cars on the same circuit lap by lap.

    Each car follows its own :class:`RaceStrategy`.  After every lap the
    simulator sorts cars by cumulative time and applies a traffic penalty
    to any car running within ``traffic_threshold_s`` of the car directly
    ahead (and that is actually slower on clean air pace).

    Parameters
    ----------
    race_simulator : RaceSimulator
        Pre-built simulator with track + vehicle already bound.
    traffic_threshold_s : float
        Gap (s) below which a following car is considered in traffic.
    traffic_penalty_s : float
        Extra time (s) added per lap spent in traffic.
    grid_gap_s : float
        Starting gap between consecutive grid positions (s).
    """

    def __init__(
        self,
        race_simulator: RaceSimulator,
        traffic_threshold_s: float = 1.0,
        traffic_penalty_s: float = 0.3,
        grid_gap_s: float = 0.3,
    ) -> None:
        self.race_sim = race_simulator
        self.traffic_threshold_s = traffic_threshold_s
        self.traffic_penalty_s = traffic_penalty_s
        self.grid_gap_s = grid_gap_s

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        entries: list[tuple[str, RaceStrategy]],
        num_laps: int,
        step_size: float = 50.0,
    ) -> MultiCarRaceResult:
        """
        Run the multi-car race.

        Parameters
        ----------
        entries : list of (name, RaceStrategy)
            Cars to simulate, in grid order (index 0 = pole).
        num_laps : int
            Race distance in laps.
        step_size : float
            Spatial integration step for the lap simulator [m].

        Returns
        -------
        MultiCarRaceResult
        """
        lap_sim = self.race_sim.lap_sim

        # Initialise per-car state
        states: list[_CarState] = []
        for grid_pos, (name, strategy) in enumerate(entries, start=1):
            gap = (grid_pos - 1) * self.grid_gap_s
            st = _CarState(
                name=name,
                strategy=strategy,
                grid_position=grid_pos,
                grid_gap_s=gap,
            )
            # Boot compound and fuel from the vehicle / strategy
            st.compound = strategy.initial_compound
            st.tyre_temp = st.compound.pit_temperature
            st.fuel_mass = lap_sim.vehicle.fuel_mass
            states.append(st)

        # ── Lap-by-lap loop ───────────────────────────────────────────
        for lap_n in range(1, num_laps + 1):

            # 1. Simulate each car's raw lap time
            lap_raw: dict[str, dict] = {}
            for st in states:
                pit = st.strategy.pit_stop_on_lap(lap_n)
                pit_loss = 0.0
                pit_flag = False

                if pit is not None:
                    pit_loss = pit.time_loss
                    st.compound = pit.new_compound
                    st.tyre_wear = 0.0
                    st.tyre_temp = st.compound.pit_temperature
                    st.laps_on_tyre = 0
                    pit_flag = True

                st.laps_on_tyre += 1

                raw = lap_sim.simulate(
                    step_size=step_size,
                    tyre_compound=st.compound,
                    initial_speed=st.speed,
                    initial_gear=st.gear,
                    initial_tyre_wear=st.tyre_wear,
                    initial_tyre_temperature=st.tyre_temp,
                    initial_fuel_mass=st.fuel_mass,
                )

                deg_penalty = st.compound.deg_s_per_lap * (st.laps_on_tyre - 1)
                raw_lap_time = raw["total_time"] + deg_penalty

                lap_raw[st.name] = {
                    "raw": raw,
                    "raw_lap_time": raw_lap_time,
                    "pit_loss": pit_loss,
                    "pit_flag": pit_flag,
                }

                # Carry state forward (before traffic adjustment)
                st.speed = raw["final_speed"]
                st.gear = raw["final_gear"]
                st.tyre_wear = raw["final_tyre_wear"]
                st.tyre_temp = raw["final_tyre_temperature"]
                st.fuel_mass = raw["final_fuel_mass"]

            # 2. Sort by current cumulative time → on-track order
            states.sort(key=lambda s: s.cumulative_time)

            # 3. Apply traffic penalties and update cumulative times
            for idx, st in enumerate(states):
                data = lap_raw[st.name]
                raw_lap_time = data["raw_lap_time"]
                pit_loss = data["pit_loss"]

                traffic_penalty = 0.0
                if idx > 0:
                    car_ahead = states[idx - 1]
                    # Gap BEFORE this lap (i.e. at start of this lap)
                    gap_to_ahead = st.cumulative_time - car_ahead.cumulative_time
                    if 0.0 <= gap_to_ahead <= self.traffic_threshold_s:
                        # Only penalise if the car ahead is on a slower pace
                        pace_ahead = lap_raw[car_ahead.name]["raw_lap_time"]
                        if pace_ahead > raw_lap_time:
                            traffic_penalty = self.traffic_penalty_s

                lap_time = raw_lap_time + pit_loss + traffic_penalty
                st.cumulative_time += lap_time
                st.total_traffic_penalty += traffic_penalty

                lap_raw[st.name]["traffic_penalty"] = traffic_penalty
                lap_raw[st.name]["lap_time"] = lap_time

            # 4. Re-sort and assign positions
            states.sort(key=lambda s: s.cumulative_time)
            leader_time = states[0].cumulative_time

            for pos, st in enumerate(states, start=1):
                data = lap_raw[st.name]
                raw_sim = data["raw"]

                car_lap = CarLapResult(
                    lap=lap_n,
                    compound=st.compound.name,
                    raw_lap_time=data["raw_lap_time"],
                    traffic_penalty=data["traffic_penalty"],
                    pit_time_loss=data["pit_loss"],
                    lap_time=data["lap_time"],
                    cumulative_time=st.cumulative_time,
                    position=pos,
                    gap_to_leader=st.cumulative_time - leader_time,
                    pit_stop=data["pit_flag"],
                    final_tyre_wear=raw_sim["final_tyre_wear"],
                    final_tyre_temperature=raw_sim["final_tyre_temperature"],
                    final_grip_multiplier=raw_sim["points"][-1]["grip_multiplier"],
                )
                st.lap_results.append(car_lap)

        # ── Build results ─────────────────────────────────────────────
        # Final order already set by last sort
        car_results: list[CarRaceResult] = []
        for final_pos, st in enumerate(states, start=1):
            car_results.append(CarRaceResult(
                name=st.name,
                strategy_name=st.strategy.name,
                grid_position=st.grid_position,
                grid_gap_s=st.grid_gap_s,
                final_position=final_pos,
                total_time=st.cumulative_time,
                total_traffic_penalty=st.total_traffic_penalty,
                laps=st.lap_results,
            ))

        return MultiCarRaceResult(
            cars=car_results,
            num_laps=num_laps,
            track=lap_sim.track.name,
        )
