# =========================================================
# Multi-Car Race Simulator
#
# Simulates N cars lap-by-lap on the same track with a
# TRACK-POSITION model: on-track passes are hard (scaled by
# the circuit's overtaking likelihood), so a faster car can be
# stuck in dirty air behind a slower one. The pit stop bypasses
# this resistance, which is what makes the UNDERCUT and OVERCUT
# the primary strategic weapons — exactly as in real racing.
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


# Reference overtaking likelihood (the calendar-average circuit). The pace
# margin a follower needs to complete an on-track pass is scaled relative to
# this: harder circuits (lower likelihood) demand a bigger advantage.
_REFERENCE_LIKELIHOOD = 0.40


class MultiCarSimulator:
    """
    Simulate multiple cars on the same circuit lap by lap with a
    track-position (undercut / overcut) model.

    Each car follows its own :class:`RaceStrategy`. After every lap the
    simulator settles the field in *track order*: a car that would pass the
    car ahead on track only succeeds if

      * it (or the car ahead) pitted this lap — the pit lane bypasses
        on-track resistance, OR
      * its clean-air pace advantage this lap exceeds the overtake margin,
        which scales inversely with the circuit's overtaking likelihood.

    Otherwise the follower is held ~``dirty_air_gap_s`` behind, losing the
    difference as a traffic (dirty-air) penalty. This makes track position
    sticky, so pitting earlier (undercut) or later (overcut) becomes the
    decisive way to gain a place.

    Parameters
    ----------
    race_simulator : RaceSimulator
        Pre-built simulator with track + vehicle already bound.
    overtaking_likelihood : float
        Circuit ease of overtaking [0–1]; lower = harder to pass on track.
    base_overtake_margin_s : float
        Pace advantage (s/lap) needed to pass on a reference circuit.
    dirty_air_gap_s : float
        Gap (s) a held-up follower sits behind the car ahead.
    grid_gap_s : float
        Starting gap between consecutive grid positions (s).
    """

    def __init__(
        self,
        race_simulator: RaceSimulator,
        overtaking_likelihood: float = _REFERENCE_LIKELIHOOD,
        base_overtake_margin_s: float = 0.5,
        dirty_air_gap_s: float = 0.7,
        grid_gap_s: float = 0.3,
    ) -> None:
        self.race_sim = race_simulator
        self.overtaking_likelihood = max(0.05, min(1.0, overtaking_likelihood))
        self.base_overtake_margin_s = base_overtake_margin_s
        self.dirty_air_gap_s = dirty_air_gap_s
        self.grid_gap_s = grid_gap_s

    @property
    def overtake_margin_s(self) -> float:
        """Pace advantage a follower needs to pass on this circuit [s/lap]."""
        return self.base_overtake_margin_s * (
            _REFERENCE_LIKELIHOOD / self.overtaking_likelihood
        )

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
        overtake_margin = self.overtake_margin_s

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

            # 2. Track order at the START of the lap (positions to defend)
            prev_order = sorted(states, key=lambda s: s.cumulative_time)

            # 3. Settle the field front-to-back with overtaking resistance.
            #    A car keeps its place unless it pits, the car ahead pits, or
            #    it has a real pace advantage — modelling sticky track position.
            new_cum: dict[str, float] = {}
            for idx, st in enumerate(prev_order):
                data = lap_raw[st.name]
                tentative = (
                    st.cumulative_time + data["raw_lap_time"] + data["pit_loss"]
                )

                if idx == 0:
                    new_cum[st.name] = tentative
                    data["traffic_penalty"] = 0.0
                    continue

                ahead = prev_order[idx - 1]
                ahead_cum = new_cum[ahead.name]

                would_pass = tentative <= ahead_cum
                pace_delta = (
                    lap_raw[ahead.name]["raw_lap_time"] - data["raw_lap_time"]
                )
                bypass = (
                    data["pit_flag"]
                    or lap_raw[ahead.name]["pit_flag"]
                    or pace_delta >= overtake_margin
                )

                if would_pass and not bypass:
                    # Held up in dirty air just behind the car ahead.
                    held_cum = ahead_cum + self.dirty_air_gap_s
                    penalty = held_cum - tentative
                    new_cum[st.name] = held_cum
                    data["traffic_penalty"] = penalty
                else:
                    new_cum[st.name] = tentative
                    data["traffic_penalty"] = 0.0

            # 4. Commit cumulative times, re-sort, assign positions
            for st in states:
                data = lap_raw[st.name]
                st.cumulative_time = new_cum[st.name]
                st.total_traffic_penalty += data["traffic_penalty"]
                data["lap_time"] = (
                    data["raw_lap_time"] + data["pit_loss"] + data["traffic_penalty"]
                )

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

        result = MultiCarRaceResult(
            cars=car_results,
            num_laps=num_laps,
            track=lap_sim.track.name,
        )
        result.overtake_events = detect_overtakes(result)
        return result


def detect_overtakes(result: MultiCarRaceResult, pit_window: int = 2) -> list:
    """
    Scan the race for position changes between cars and classify each as an
    **undercut**, **overcut**, or **on-track pass**.

    A position swap on lap L is attributed to the pit cycle (undercut/overcut)
    if either car involved pitted within ``pit_window`` laps of L; otherwise it
    is recorded as an on-track pass. Returns a list of ``OvertakeEvent``.
    """
    cars = result.cars
    # Map lap → {car_name: (position, pitted_recently)}
    n_laps = result.num_laps
    # Build per-car lap-indexed position and recent-pit lookup
    pos_by_lap: dict[str, dict[int, int]] = {}
    pit_laps: dict[str, set[int]] = {}
    for c in cars:
        pos_by_lap[c.name] = {lr.lap: lr.position for lr in c.laps}
        pit_laps[c.name] = {lr.lap for lr in c.laps if lr.pit_stop}

    def pitted_near(name: str, lap: int) -> bool:
        return any(abs(p - lap) <= pit_window for p in pit_laps[name])

    events = []
    names = [c.name for c in cars]
    for lap in range(2, n_laps + 1):
        for a in names:
            for b in names:
                if a >= b:
                    continue
                pa_prev = pos_by_lap[a].get(lap - 1)
                pb_prev = pos_by_lap[b].get(lap - 1)
                pa_now = pos_by_lap[a].get(lap)
                pb_now = pos_by_lap[b].get(lap)
                if None in (pa_prev, pb_prev, pa_now, pb_now):
                    continue
                # Did a and b swap relative order this lap?
                was_ahead = pa_prev < pb_prev
                now_ahead = pa_now < pb_now
                if was_ahead == now_ahead:
                    continue
                gainer, loser = (a, b) if now_ahead else (b, a)
                if pitted_near(gainer, lap):
                    kind = "undercut"
                elif pitted_near(loser, lap):
                    kind = "overcut"
                else:
                    kind = "on-track pass"
                events.append(OvertakeEvent(
                    lap=lap, gainer=gainer, loser=loser, kind=kind,
                    new_position=pos_by_lap[gainer].get(lap),
                ))
    return events


@dataclass
class OvertakeEvent:
    """A position change between two cars, classified by cause."""
    lap: int
    gainer: str
    loser: str
    kind: str            # "undercut" | "overcut" | "on-track pass"
    new_position: int
