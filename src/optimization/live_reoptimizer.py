# =========================================================
# Live race re-optimiser
#
# The deterministic DP plans the whole race up front. But strategy in F1
# is made IN the race, reacting to events: a Safety Car at lap 23, rain
# arriving, a rival's undercut. This module answers the real pit-wall
# question — "given where we are RIGHT NOW, what is the optimal move?".
#
# It re-optimises only the REMAINING race from the current state (current
# compound + tyre age + fuel), so it correctly handles a part-worn tyre.
# Under a Safety Car the in-lap pit loss is discounted (the whole field is
# slow), which is exactly what makes a well-timed SC stop so valuable.
#
# Reuses the existing lap simulator; it does NOT touch the calibrated model.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field

from src.models.tyre import TyreCompound
from src.models.strategy import RaceResult


@dataclass
class RaceState:
    """Snapshot of the car mid-race (after completing ``lap``)."""
    lap: int                       # last completed lap
    compound: TyreCompound         # tyre currently fitted
    tyre_age: int                  # laps done on the current set
    tyre_wear: float               # current wear [0-1+]
    tyre_temperature: float
    fuel_mass: float
    used_compounds: set[str]       # compound NAMES used so far (two-compound rule)
    speed: float = 30.0
    gear: int = 6

    @classmethod
    def from_result(cls, result: RaceResult, lap: int) -> "RaceState":
        """Reconstruct the state after ``lap`` from a deterministic race result."""
        lr = result.laps[lap - 1]
        # Tyre age: laps since the last pit at or before this lap.
        age = 0
        used: set[str] = set()
        for i in range(lap):
            li = result.laps[i]
            used.add(li.compound)
            age = 1 if li.pit_stop else age + 1
        return cls(
            lap=lap, compound=_compound_of(result, lap), tyre_age=age,
            tyre_wear=lr.final_tyre_wear, tyre_temperature=lr.final_tyre_temperature,
            fuel_mass=lr.final_fuel_mass, used_compounds=used,
            speed=lr.final_speed, gear=lr.final_gear,
        )


def _compound_of(result: RaceResult, lap: int) -> TyreCompound:
    # The LapResult stores compound by name; map back via the strategy stints.
    name = result.laps[lap - 1].compound
    for st in result.stints:
        if st.start_lap <= lap <= st.end_lap:
            return st.compound
    # Fallback: search any stint whose compound name matches.
    for st in result.stints:
        if st.compound.name == name:
            return st.compound
    raise ValueError(f"Cannot resolve compound for lap {lap}")


@dataclass
class LiveOption:
    """One candidate continuation of the race from the current state."""
    label: str
    pit_now: bool
    remaining_pits: list[tuple[int, str]]   # (lap, compound name)
    remaining_time: float                   # time from now to the finish [s]
    delta_vs_best: float = 0.0


class LiveReoptimizer:
    """
    Re-optimise the remaining race from a given state, optionally under a
    Safety Car (which discounts the cost of pitting now).
    """

    # Lap-time inflation while the regime is active (matches the Monte Carlo).
    _SC_FACTOR  = 1.60
    _VSC_FACTOR = 1.38

    def __init__(
        self,
        lap_sim,
        num_laps: int,
        pit_loss: float,
        compounds: list[TyreCompound],
        min_stint_laps: int = 5,
        require_two_compounds: bool = True,
        step_size: float = 50.0,
        weather=None,
    ) -> None:
        self.lap_sim = lap_sim
        self.num_laps = num_laps
        self.pit_loss = pit_loss
        self.compounds = compounds
        self.min_stint = min_stint_laps
        self.require_two = require_two_compounds
        self.step = step_size
        # Optional WeatherModel: the remaining race must be simulated under the
        # SAME per-lap track wetness as the real race, otherwise the optimiser
        # is weather-blind (e.g. it would fit slicks on a still-wet track).
        self.weather = weather

    # ------------------------------------------------------------------ #
    # Remaining-race simulation                                            #
    # ------------------------------------------------------------------ #

    def _sim_remaining(
        self,
        state: RaceState,
        pits: list[tuple[int, TyreCompound]],
        sc_window: set[int],
        regime_factor: float,
    ) -> float:
        """
        Time from the next lap to the finish, given the remaining pit plan.

        ``sc_window`` is the set of laps run under the active Safety Car. Those
        laps are inflated by ``regime_factor`` for EVERY option (the SC is a
        track state, it slows the whole field regardless of strategy), and a
        pit taken on one of those laps gets its pit loss discounted (the field
        is slow, so the pit-lane differential shrinks). The strategic value of
        an SC therefore lives entirely in the pit discount.
        """
        pit_map = {lap: comp for lap, comp in pits}
        compound = state.compound
        wear = state.tyre_wear
        temp = state.tyre_temperature
        fuel = state.fuel_mass
        speed, gear = state.speed, state.gear
        age = state.tyre_age
        total = 0.0

        for lap in range(state.lap + 1, self.num_laps + 1):
            in_sc = lap in sc_window
            pit_here = pit_map.get(lap)
            if pit_here is not None:
                compound = pit_here
                wear = 0.0
                temp = compound.pit_temperature
                age = 0
                total += (self.pit_loss / regime_factor) if in_sc else self.pit_loss
            age += 1
            # Match the real per-lap track wetness (dynamic weather), so the
            # remaining-race pace and the compound choice stay weather-correct.
            if self.weather is not None:
                self.lap_sim.track_wetness = self.weather.wetness(lap)
            raw = self.lap_sim.simulate(
                step_size=self.step, tyre_compound=compound,
                initial_speed=speed, initial_gear=gear,
                initial_tyre_wear=wear, initial_tyre_temperature=temp,
                initial_fuel_mass=fuel,
            )
            lap_t = raw["total_time"] + compound.deg_s_per_lap * (age - 1)
            if in_sc:                      # SC slows every option equally
                lap_t *= regime_factor
            total += lap_t
            wear = raw["final_tyre_wear"]
            temp = raw["final_tyre_temperature"]
            fuel = raw["final_fuel_mass"]
            speed = raw["final_speed"]
            gear = raw["final_gear"]
        return total

    # ------------------------------------------------------------------ #
    # Decision                                                             #
    # ------------------------------------------------------------------ #

    def _legal(self, state: RaceState, remaining_pits: list) -> bool:
        """Two-compound rule over the whole race (waived when wet)."""
        if not self.require_two:
            return True
        used = set(state.used_compounds) | {c.name for _, c in remaining_pits}
        return len(used) >= 2

    def decide(
        self,
        state: RaceState,
        regime: str = "green",
        sc_duration: int = 4,
        max_remaining_stops: int = 2,
    ) -> list[LiveOption]:
        """
        Rank continuations of the race from ``state``.

        ``regime`` in {"green","vsc","sc"}: under sc/vsc a window of
        ``sc_duration`` laps from now is run slower for ALL options, and any
        pit taken inside that window is discounted.
        """
        factor = (self._SC_FACTOR if regime == "sc"
                  else self._VSC_FACTOR if regime == "vsc" else 1.0)
        now_lap = state.lap + 1
        remaining = self.num_laps - state.lap
        sc_window = (set(range(now_lap, now_lap + sc_duration))
                     if regime != "green" else set())
        options: list[LiveOption] = []

        cand_laps = list(range(now_lap, self.num_laps - self.min_stint + 2))

        # --- 0 remaining stops: run current tyre to the end ----------------
        if remaining >= self.min_stint and self._legal(state, []):
            t = self._sim_remaining(state, [], sc_window, factor)
            options.append(LiveOption("STAY OUT — no more stops", False, [], t))

        # --- 1 remaining stop ----------------------------------------------
        for c in self.compounds:
            for p in cand_laps:
                if (p - now_lap) < 0 or (self.num_laps - p + 1) < self.min_stint:
                    continue
                plan = [(p, c)]
                if not self._legal(state, plan):
                    continue
                t = self._sim_remaining(state, plan, sc_window, factor)
                pit_now = (p == now_lap)
                tag = f"PIT NOW → {c.name}" if pit_now else f"pit L{p} → {c.name}"
                options.append(LiveOption(tag, pit_now, [(p, c.name)], t))

        # --- 2 remaining stops (coarser sampling for speed) ----------------
        if max_remaining_stops >= 2:
            step_laps = max(2, self.min_stint)
            for c1 in self.compounds:
                for c2 in self.compounds:
                    for p1 in cand_laps[::step_laps]:
                        for p2 in range(p1 + self.min_stint,
                                         self.num_laps - self.min_stint + 2, step_laps):
                            plan = [(p1, c1), (p2, c2)]
                            if not self._legal(state, plan):
                                continue
                            t = self._sim_remaining(state, plan, sc_window, factor)
                            pit_now = (p1 == now_lap)
                            tag = (f"PIT NOW → {c1.name}, then L{p2} → {c2.name}"
                                   if pit_now else
                                   f"pit L{p1} → {c1.name}, L{p2} → {c2.name}")
                            options.append(LiveOption(tag, pit_now,
                                                      [(p1, c1.name), (p2, c2.name)], t))

        options.sort(key=lambda o: o.remaining_time)
        best = options[0].remaining_time if options else 0.0
        for o in options:
            o.delta_vs_best = o.remaining_time - best
        return options
