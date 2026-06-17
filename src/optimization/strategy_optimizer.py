# =========================================================
# Strategy Optimizer
#
# Advanced strategy optimization via Dynamic Programming.
#
# DPStrategyOptimizer   — exact optimal solution in O(L² · C · S)
# GeneticStrategyOptimizer — placeholder (not yet implemented)
# =========================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from src.models.tyre import TyreCompound
from src.models.strategy import PitStop, RaceResult, RaceStrategy
from src.simulation.race_simulator import RaceSimulator


# ------------------------------------------------------------------ #
# Internal DP types                                                   #
# ------------------------------------------------------------------ #

class _StintKey(NamedTuple):
    compound_name: str
    start_lap: int      # 1-based, lap on which this stint begins
    length: int         # number of laps in the stint


@dataclass
class _DPNode:
    """Best total race time reachable at (lap, compound_idx, stops_used)."""
    cost: float = math.inf
    # Backtracking: the stint that led to this node
    prev_lap: int = -1
    prev_compound_idx: int = -1
    prev_stops: int = -1


# ------------------------------------------------------------------ #
# Dynamic Programming optimizer                                       #
# ------------------------------------------------------------------ #

class DPStrategyOptimizer:
    """
    Optimal pit-stop strategy finder using Dynamic Programming.

    Complexity
    ----------
    Brute-force (strategy_search) simulates every full race:
        O(L^stops · C^stops) full-race simulations

    DP simulates each possible stint once, then solves the
    combination problem analytically:
        - Stint table build: O(C · L²/2) fresh-tyre stint simulations
        - DP solve:          O(L² · C² · S)  (pure arithmetic)
        Total simulator calls ≈ C · L²/2  (no repeated work)

    For Monza (53 laps, 3 compounds, max 2 stops):
        Brute-force: ~4 700 full-race sims   (~1 h Python)
        DP:          ~ 1800 stint entries     (~2-5 min)
        Speedup:     significant, while preserving fresh-tyre pit stints

    Parameters
    ----------
    race_simulator : RaceSimulator
        Pre-built simulator (track + vehicle already bound).
    min_stint_laps : int
        Minimum laps per stint (default 10 for 50+ lap races).
    verbose : bool
        Print progress during the stint-table build phase.
    """

    def __init__(
        self,
        race_simulator: RaceSimulator,
        min_stint_laps: int = 10,
        verbose: bool = True,
    ) -> None:
        self.race_sim       = race_simulator
        self.min_stint_laps = min_stint_laps
        self.verbose        = verbose

        # Populated by _build_stint_table()
        self._stint_times:   dict[_StintKey, float] = {}
        self._stint_wear:    dict[_StintKey, float] = {}   # final wear (info only)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        num_laps: int,
        compounds: list[TyreCompound],
        pit_loss: float = 22.0,
        max_stops: int = 2,
        require_two_compounds: bool = True,
        step_size: float = 5.0,
    ) -> list[RaceResult]:
        """
        Find the optimal race strategy using Dynamic Programming.

        Returns a list of RaceResult sorted by total time (best first),
        one per distinct compound sequence — mirroring the interface of
        generate_and_simulate() so callers can swap the two transparently.

        Parameters
        ----------
        num_laps : int
            Total race distance in laps.
        compounds : list[TyreCompound]
            Tyre compounds available for this event.
        pit_loss : float
            Fixed time penalty per pit stop [s].
        max_stops : int
            Maximum number of pit stops (1 or 2).
        require_two_compounds : bool
            Enforce the FIA two-compound rule.
        step_size : float
            Spatial integration step [m] passed to the simulator.

        Returns
        -------
        list[RaceResult]
            Sorted results (fastest first). Typically one result per
            valid compound sequence (6 for 1-stop, 12 for 2-stop with
            3 compounds).
        """
        if num_laps < 1:
            raise ValueError("num_laps must be >= 1")
        if max_stops < 1 or max_stops > 2:
            raise ValueError("max_stops must be 1 or 2")

        # Phase 1 — build the stint time table
        self._build_stint_table(
            num_laps=num_laps,
            compounds=compounds,
            step_size=step_size,
        )

        # Phase 2 — DP over all valid compound sequences
        results: list[RaceResult] = []

        sequences = _valid_sequences(
            compounds=compounds,
            max_stops=max_stops,
            require_two_compounds=require_two_compounds,
        )

        if self.verbose:
            print(
                f"\nDP solve: {len(sequences)} compound sequences × "
                f"{num_laps} laps ...",
                flush=True,
            )

        for seq in sequences:
            result = self._solve_sequence(
                sequence=seq,
                num_laps=num_laps,
                pit_loss=pit_loss,
                step_size=step_size,
            )
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.total_time)

        if self.verbose:
            if results:
                best = results[0]
                print(
                    f"DP best: {best.strategy}  "
                    f"total={best.total_time:.3f} s",
                    flush=True,
                )

        return results

    # ------------------------------------------------------------------ #
    # Phase 1 — stint table                                               #
    # ------------------------------------------------------------------ #

    def _build_stint_table(
        self,
        num_laps: int,
        compounds: list[TyreCompound],
        step_size: float,
        pit_window_step: int = 1,
    ) -> None:
        """
        Pre-compute stint times with fresh tyres at each possible stint start.

        A stint beginning after a pit stop must not inherit tyre wear from an
        earlier lap on the same compound. For each valid (compound, start_lap)
        pair, we simulate one fresh-tyre trajectory from the estimated fuel
        load at that race lap and use prefix sums within that trajectory.
        """
        self._stint_times.clear()
        self._stint_wear.clear()

        vehicle   = self.race_sim.lap_sim.vehicle
        lap_sim   = self.race_sim.lap_sim

        # Candidate pit laps for the DP solve (sampled every pit_window_step)
        pit_laps = list(range(
            1 + self.min_stint_laps,
            num_laps - self.min_stint_laps + 1,
            pit_window_step,
        ))
        last_valid = num_laps - self.min_stint_laps
        if last_valid not in pit_laps:
            pit_laps.append(last_valid)
        valid_starts = sorted({1} | {p + 1 for p in pit_laps})

        if self.verbose:
            total_laps = sum(
                num_laps - start + 1
                for _compound in compounds
                for start in valid_starts
            )
            print(
                f"\nBuilding fresh-tyre stint table: "
                f"{len(compounds)} compounds × {len(valid_starts)} starts "
                f"= {total_laps} lap sims...",
                flush=True,
            )

        t0 = time.monotonic()

        done = 0
        fuel_burn_per_lap = (
            vehicle.fuel_consumption_per_km * self.race_sim.lap_sim.track.total_length / 1000.0
        )

        for compound in compounds:
            for start in valid_starts:
                remaining = num_laps - start + 1
                if remaining < self.min_stint_laps:
                    continue

                # Fresh tyre set fitted at this stint start. Fuel is estimated
                # from race distance already completed; the trajectory then
                # burns fuel normally through the lap simulator.
                speed     = 1.0
                gear      = 1
                tyre_wear = 0.0
                tyre_temp = compound.pit_temperature
                fuel_mass = max(0.0, vehicle.fuel_mass - fuel_burn_per_lap * (start - 1))

                cumulative_time: list[float] = [0.0]
                lap_wear:         list[float] = [0.0]

                weather = getattr(self.race_sim, "weather", None)

                for lap_idx in range(remaining):
                    # Dynamic weather (Level B): the absolute race lap is
                    # start + lap_idx, so the stint table is built under the
                    # forecast and the DP optimises pit laps accordingly.
                    if weather is not None:
                        lap_sim.track_wetness = weather.wetness(start + lap_idx)

                    result = lap_sim.simulate(
                        step_size=step_size,
                        tyre_compound=compound,
                        initial_speed=speed,
                        initial_gear=gear,
                        initial_tyre_wear=tyre_wear,
                        initial_tyre_temperature=tyre_temp,
                        initial_fuel_mass=fuel_mass,
                    )
                    # Empirical degradation overlay — must match RaceSimulator,
                    # which adds deg_s_per_lap × (laps_on_tyre − 1) on top of the
                    # physics lap time. Here lap_idx == laps_on_tyre − 1 (fresh
                    # tyre at lap_idx 0). Without this the DP optimises pit laps
                    # against a different objective than the one the race is
                    # ultimately scored with.
                    deg_penalty = compound.deg_s_per_lap * lap_idx
                    cumulative_time.append(
                        cumulative_time[-1] + result["total_time"] + deg_penalty
                    )
                    lap_wear.append(result["final_tyre_wear"])

                    speed     = result["final_speed"]
                    gear      = result["final_gear"]
                    tyre_wear = result["final_tyre_wear"]
                    tyre_temp = result["final_tyre_temperature"]
                    fuel_mass = result["final_fuel_mass"]

                    if self.verbose:
                        done += 1
                        _print_progress(done, total_laps, t0)

                lengths = list(range(self.min_stint_laps, remaining + 1, pit_window_step))
                if remaining not in lengths:
                    lengths.append(remaining)
                for length in lengths:
                    if length > remaining:
                        continue
                    t = cumulative_time[length]
                    w = lap_wear[length]
                    self._stint_times[_StintKey(compound.name, start, length)] = t
                    self._stint_wear[_StintKey(compound.name, start, length)]  = w

        if self.verbose:
            n = len(self._stint_times)
            print(
                f"\nStint table ready: {n} entries in "
                f"{time.monotonic() - t0:.1f} s",
                flush=True,
            )

    def _simulate_stint(self, *args, **kwargs):
        """Superseded by fresh-tyre trajectories in _build_stint_table."""
        raise NotImplementedError("Use _build_stint_table instead.")

    # ------------------------------------------------------------------ #
    # Phase 2 — DP solve for one compound sequence                        #
    # ------------------------------------------------------------------ #

    def _solve_sequence(
        self,
        sequence: list[TyreCompound],
        num_laps: int,
        pit_loss: float,
        step_size: float,
    ) -> RaceResult | None:
        """
        Given a fixed compound sequence (e.g. [Soft, Hard] for a 1-stop),
        find the optimal pit lap(s) using DP.

        State: (lap after which we pit, stint index completed so far)
        Value: minimum cumulative raw race time up to that point

        For a k-stop race we have k+1 stints. We iterate over all valid
        split points using the pre-computed stint table.
        """
        num_stints = len(sequence)   # 2 for 1-stop, 3 for 2-stop

        if num_stints == 2:
            return self._solve_one_stop(sequence, num_laps, pit_loss, step_size)
        elif num_stints == 3:
            return self._solve_two_stop(sequence, num_laps, pit_loss, step_size)
        else:
            return None   # not supported

    def _solve_one_stop(
        self,
        sequence: list[TyreCompound],
        num_laps: int,
        pit_loss: float,
        step_size: float,
    ) -> RaceResult | None:
        """
        1-stop DP: find pit_lap p (1-based) that minimises
            time(c1, start=1, length=p-1)            + pit_loss
          + time(c2, start=p, length=num_laps-p+1)

        PitStop(lap=p) fires before lap p → stint1 covers laps 1..p-1.
        """
        c1, c2 = sequence
        best_time = math.inf
        best_pit  = -1

        for p in range(
            self.min_stint_laps + 1,             # stint1 = p-1 >= min_stint
            num_laps - self.min_stint_laps + 2,  # stint2 >= min_stint
        ):
            length1 = p - 1
            length2 = num_laps - p + 1
            stint1  = self._get_stint_time(c1, 1, length1)
            stint2  = self._get_stint_time(c2, p, length2)
            if stint1 is None or stint2 is None:
                continue
            total = stint1 + pit_loss + stint2
            if total < best_time:
                best_time = total
                best_pit  = p

        if best_pit == -1:
            return None

        strategy = RaceStrategy(
            name=f"DP {c1.name}-{c2.name} L{best_pit}",
            initial_compound=c1,
            pit_stops=[PitStop(best_pit, c2, pit_loss)],
        )
        return self.race_sim.simulate(num_laps, strategy, step_size)

    def _solve_two_stop(
        self,
        sequence: list[TyreCompound],
        num_laps: int,
        pit_loss: float,
        step_size: float,
    ) -> RaceResult | None:
        """
        2-stop DP: find pit laps (p1, p2) minimising
            time(c1, start=1,  length=p1-1)      + pit_loss
          + time(c2, start=p1, length=p2-p1)     + pit_loss
          + time(c3, start=p2, length=num_laps-p2+1)
        """
        c1, c2, c3 = sequence
        best_time = math.inf
        best_p1   = -1
        best_p2   = -1

        for p1 in range(
            self.min_stint_laps + 1,
            num_laps - 2 * self.min_stint_laps + 2,
        ):
            stint1 = self._get_stint_time(c1, 1, p1 - 1)
            if stint1 is None:
                continue

            for p2 in range(
                p1 + self.min_stint_laps,
                num_laps - self.min_stint_laps + 2,
            ):
                stint2 = self._get_stint_time(c2, p1, p2 - p1)
                stint3 = self._get_stint_time(c3, p2, num_laps - p2 + 1)
                if stint2 is None or stint3 is None:
                    continue
                total = stint1 + stint2 + stint3 + 2 * pit_loss
                if total < best_time:
                    best_time = total
                    best_p1   = p1
                    best_p2   = p2

        if best_p1 == -1:
            return None

        strategy = RaceStrategy(
            name=f"DP {c1.name}-{c2.name}-{c3.name} L{best_p1}/L{best_p2}",
            initial_compound=c1,
            pit_stops=[
                PitStop(best_p1, c2, pit_loss),
                PitStop(best_p2, c3, pit_loss),
            ],
        )
        return self.race_sim.simulate(num_laps, strategy, step_size)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_stint_time(
        self,
        compound: TyreCompound,
        start_lap: int,
        length: int,
    ) -> float | None:
        """Look up a pre-computed stint time; return None if not in table."""
        return self._stint_times.get(_StintKey(compound.name, start_lap, length))


# ------------------------------------------------------------------ #
# Helpers shared between optimizer and caller                         #
# ------------------------------------------------------------------ #

def _valid_sequences(
    compounds: list[TyreCompound],
    max_stops: int,
    require_two_compounds: bool,
) -> list[list[TyreCompound]]:
    """
    Return all valid compound sequences for up to max_stops pit stops.
    A sequence has (max_stops + 1) elements — one per stint.
    """
    sequences: list[list[TyreCompound]] = []

    if max_stops >= 1:
        for c1 in compounds:
            for c2 in compounds:
                if c1.name == c2.name:
                    continue
                if require_two_compounds and len({c1.name, c2.name}) < 2:
                    continue
                sequences.append([c1, c2])

    if max_stops >= 2:
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    seq = [c1, c2, c3]
                    # No two consecutive same compounds
                    if c1.name == c2.name or c2.name == c3.name:
                        continue
                    if require_two_compounds and len({c.name for c in seq}) < 2:
                        continue
                    sequences.append(seq)

    return sequences


def _print_progress(current: int, total: int, t0: float) -> None:
    pct     = current / total
    filled  = int(40 * pct)
    bar     = "█" * filled + "░" * (40 - filled)
    elapsed = time.monotonic() - t0
    eta     = (elapsed / current) * (total - current) if current else 0.0
    print(
        f"\r  [{bar}] {current:>5}/{total}"
        f"  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s",
        end="",
        flush=True,
    )


# ------------------------------------------------------------------ #
# Genetic Algorithm — placeholder                                     #
# ------------------------------------------------------------------ #

class GeneticStrategyOptimizer:
    """
    Genetic-algorithm strategy optimizer.

    NOT YET IMPLEMENTED.
    """

    def __init__(
        self,
        race_simulator: RaceSimulator,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.05,
    ) -> None:
        self.race_simulator  = race_simulator
        self.population_size = population_size
        self.generations     = generations
        self.mutation_rate   = mutation_rate

    def optimize(
        self,
        num_laps: int,
        compounds: list[TyreCompound],
        pit_loss: float = 22.0,
        max_stops: int = 2,
        step_size: float = 5.0,
    ) -> RaceResult:
        raise NotImplementedError(
            "GeneticStrategyOptimizer is not yet implemented. "
            "Use DPStrategyOptimizer or strategy_search.generate_and_simulate()."
        )
    
