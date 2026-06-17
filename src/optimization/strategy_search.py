# =========================================================
# Strategy Search
#
# Generates all valid candidate strategies (brute-force
# combinatorial enumeration) and runs them through the
# race simulator to produce a ranked pool of RaceResults.
#
# This is intentionally separate from strategy_optimizer.py,
# which will house more advanced search methods (DP, genetic
# algorithms, etc.) in the future.
# =========================================================

from __future__ import annotations

import time

from src.models.tyre import TyreCompound
from src.models.strategy import PitStop, RaceStrategy, RaceResult
from src.simulation.race_simulator import RaceSimulator


# ------------------------------------------------------------------ #
# Candidate generation                                                #
# ------------------------------------------------------------------ #

def generate_candidate_strategies(
    num_laps: int,
    compounds: list[TyreCompound],
    pit_loss: float = 22.0,
    min_stint_laps: int = 2,
    max_stops: int = 2,
    require_two_compounds: bool = True,
) -> list[RaceStrategy]:
    """
    Enumerate all valid race strategies up to max_stops pit stops.

    Validity rules
    --------------
    - No two consecutive stints on the same compound.
    - If require_two_compounds=True, at least two distinct compounds
      must appear across the whole race.
    - Every stint must be at least min_stint_laps long.
    """
    strategies: list[RaceStrategy] = []

    if max_stops >= 1:
        strategies.extend(
            _one_stop_strategies(
                num_laps=num_laps,
                compounds=compounds,
                pit_loss=pit_loss,
                min_stint_laps=min_stint_laps,
                require_two_compounds=require_two_compounds,
            )
        )

    if max_stops >= 2:
        strategies.extend(
            _two_stop_strategies(
                num_laps=num_laps,
                compounds=compounds,
                pit_loss=pit_loss,
                min_stint_laps=min_stint_laps,
                require_two_compounds=require_two_compounds,
            )
        )

    return strategies


# ------------------------------------------------------------------ #
# Strategy pool simulation                                            #
# ------------------------------------------------------------------ #

def simulate_strategy_pool(
    race_simulator: RaceSimulator,
    strategies: list[RaceStrategy],
    num_laps: int,
    step_size: float = 5.0,
    max_candidates: int | None = None,
    verbose: bool = True,
) -> list[RaceResult]:
    """
    Run every strategy through the race simulator and return results
    sorted by total race time (fastest first).

    Parameters
    ----------
    max_candidates : int | None
        If set, sample this many strategies evenly from the full pool
        before simulating. Useful to keep runtime under control on
        long races (e.g. 53 laps brute-force -> ~4 700 strategies).
        Recommended: 200-500 for interactive runs; None for overnight.
        Sampling is deterministic (evenly spaced indices) so all
        pit-window positions are always represented.
    verbose : bool
        Print a live progress bar to stdout.
    """
    pool  = _sample_strategies(strategies, max_candidates)
    total = len(pool)

    if verbose:
        skipped = len(strategies) - total
        suffix  = f" (sampled from {len(strategies)})" if skipped else ""
        print(f"\nSimulating {total} strategies{suffix}...", flush=True)

    results: list[RaceResult] = []
    t0 = time.monotonic()

    for i, strategy in enumerate(pool, start=1):
        results.append(
            race_simulator.simulate(
                num_laps=num_laps,
                strategy=strategy,
                step_size=step_size,
            )
        )
        if verbose:
            _print_progress(i, total, t0)

    if verbose:
        print(f"\nDone in {time.monotonic() - t0:.1f} s", flush=True)

    return sorted(results, key=lambda r: r.total_time)


def generate_and_simulate(
    race_simulator: RaceSimulator,
    num_laps: int,
    compounds: list[TyreCompound],
    pit_loss: float = 22.0,
    min_stint_laps: int = 2,
    max_stops: int = 2,
    require_two_compounds: bool = True,
    step_size: float = 5.0,
    max_candidates: int | None = None,
    verbose: bool = True,
) -> list[RaceResult]:
    """
    Convenience wrapper: generate candidates + simulate in one call.

    Returns sorted RaceResults (fastest first).
    """
    strategies = generate_candidate_strategies(
        num_laps=num_laps,
        compounds=compounds,
        pit_loss=pit_loss,
        min_stint_laps=min_stint_laps,
        max_stops=max_stops,
        require_two_compounds=require_two_compounds,
    )

    return simulate_strategy_pool(
        race_simulator=race_simulator,
        strategies=strategies,
        num_laps=num_laps,
        step_size=step_size,
        max_candidates=max_candidates,
        verbose=verbose,
    )


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _sample_strategies(
    strategies: list[RaceStrategy],
    max_candidates: int | None,
) -> list[RaceStrategy]:
    """
    Return up to max_candidates strategies sampled at evenly-spaced
    indices so the full pit-window range is always represented.
    Returns the full list unchanged if max_candidates is None or
    larger than the pool.
    """
    if max_candidates is None or max_candidates >= len(strategies):
        return strategies

    n    = len(strategies)
    step = n / max_candidates
    seen: set[int] = set()
    for k in range(max_candidates):
        seen.add(int(k * step))
    return [strategies[i] for i in sorted(seen)]


def _print_progress(current: int, total: int, t0: float) -> None:
    """Compact inline progress bar printed to stdout."""
    pct     = current / total
    filled  = int(40 * pct)
    bar     = "█" * filled + "░" * (40 - filled)
    elapsed = time.monotonic() - t0
    eta     = (elapsed / current) * (total - current) if current else 0.0
    print(
        f"\r  [{bar}] {current:>4}/{total}"
        f"  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s",
        end="",
        flush=True,
    )


def _is_valid_sequence(
    sequence: list[TyreCompound],
    require_two_compounds: bool,
) -> bool:
    for i in range(len(sequence) - 1):
        if sequence[i].name == sequence[i + 1].name:
            return False
    if require_two_compounds:
        if len({c.name for c in sequence}) < 2:
            return False
    return True


def _one_stop_strategies(
    num_laps: int,
    compounds: list[TyreCompound],
    pit_loss: float,
    min_stint_laps: int,
    require_two_compounds: bool,
) -> list[RaceStrategy]:
    strategies: list[RaceStrategy] = []
    for c1 in compounds:
        for c2 in compounds:
            if not _is_valid_sequence([c1, c2], require_two_compounds):
                continue
            for pit_lap in range(
                1 + min_stint_laps,
                num_laps - min_stint_laps + 2,
            ):
                strategies.append(
                    RaceStrategy(
                        name=f"{c1.name}-{c2.name} L{pit_lap}",
                        initial_compound=c1,
                        pit_stops=[PitStop(pit_lap, c2, pit_loss)],
                    )
                )
    return strategies


def _two_stop_strategies(
    num_laps: int,
    compounds: list[TyreCompound],
    pit_loss: float,
    min_stint_laps: int,
    require_two_compounds: bool,
) -> list[RaceStrategy]:
    strategies: list[RaceStrategy] = []
    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                if not _is_valid_sequence([c1, c2, c3], require_two_compounds):
                    continue
                for pit1 in range(
                    1 + min_stint_laps,
                    num_laps - 2 * min_stint_laps + 2,
                ):
                    for pit2 in range(
                        pit1 + min_stint_laps,
                        num_laps - min_stint_laps + 2,
                    ):
                        strategies.append(
                            RaceStrategy(
                                name=(
                                    f"{c1.name}-{c2.name}-{c3.name} "
                                    f"L{pit1}/L{pit2}"
                                ),
                                initial_compound=c1,
                                pit_stops=[
                                    PitStop(pit1, c2, pit_loss),
                                    PitStop(pit2, c3, pit_loss),
                                ],
                            )
                        )
    return strategies

