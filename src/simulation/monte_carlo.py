# =========================================================
# Monte Carlo race simulator
#
# Turns a single deterministic strategy outcome into a *distribution*
# of race times by sampling the stochastic events that dominate real
# F1 strategy: Safety Cars and Virtual Safety Cars.
#
# Why this matters (and why it is not just "physics again"):
#   The deterministic optimiser gives ONE optimal strategy for ONE
#   green-flag race. Real strategy decisions are made under uncertainty —
#   a Safety Car at the wrong moment can ruin a 1-stop and gift a 2-stop.
#   This layer answers "how ROBUST is each strategy?", not just "which is
#   fastest on paper". Output: P5 / P50 / P95 and a win probability.
#
# Performance: the physics is NOT re-run. Each sample perturbs the
# already-computed per-lap times, so 1000s of races cost milliseconds.
# This is exactly how production strategy tools run their Monte Carlo.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.models.strategy import RaceResult


# Regime codes
_GREEN = 0
_VSC   = 1
_SC    = 2


def _tyre_age_per_lap(result) -> np.ndarray:
    """
    Tyre age (laps on the current set) for each race lap, reconstructed from
    the pit flags. Matches RaceSimulator.laps_on_tyre: the pit lap itself is
    age 1, then it increments until the next stop.
    """
    ages = []
    a = 0
    for lr in result.laps:
        a = 1 if lr.pit_stop else a + 1
        ages.append(a)
    return np.array(ages, dtype=float)


@dataclass(frozen=True)
class SafetyCarParams:
    """Stochastic race-control parameters (per circuit)."""
    sc_prob_per_lap: float          # P(a Safety Car starts on a given green lap)
    vsc_prob_per_lap: float         # P(a Virtual Safety Car starts on a given green lap)
    avg_sc_duration_laps: int = 4   # how long a full SC lasts
    avg_vsc_duration_laps: int = 2  # how long a VSC lasts

    # Lap-time inflation while the regime is active (vs green-flag pace).
    sc_lap_factor: float = 1.60     # bunched up behind the Safety Car
    vsc_lap_factor: float = 1.38    # running to a delta time under VSC

    # Lap-to-lap execution noise (driver/track variability), 1σ in seconds.
    pace_noise_s: float = 0.12


@dataclass
class StrategyDistribution:
    """Monte Carlo outcome distribution for one strategy."""
    name: str
    deterministic_total: float      # green-flag race time (no events)
    mean: float
    p5: float
    p50: float
    p95: float
    std: float
    win_probability: float          # fraction of scenarios where this strategy wins
    sc_exposure_pct: float          # % of races that saw >=1 full Safety Car
    neutralisation_pct: float       # % of races with ANY neutralisation (SC or VSC)
    samples: np.ndarray = field(repr=False, default=None)

    @property
    def robustness_s(self) -> float:
        """Spread P95−P5: smaller = more robust to race-control events."""
        return self.p95 - self.p5


class MonteCarloRaceSimulator:
    """
    Evaluate strategies under stochastic Safety Car / VSC events.

    Uses *common random numbers*: every strategy is scored against the
    SAME set of sampled race scenarios, so the win-probability comparison
    is fair (paired) and low-variance.
    """

    def __init__(
        self,
        params: SafetyCarParams,
        num_samples: int = 1000,
        seed: int | None = 42,
    ) -> None:
        self.params = params
        self.num_samples = num_samples
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Scenario sampling                                                    #
    # ------------------------------------------------------------------ #

    def _sample_regime(self, num_laps: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a per-lap race-control regime array (green / VSC / SC).

        Sequential scan: on each green lap an SC or VSC may start (Bernoulli
        per lap), then occupies the next `duration` laps. SC takes priority.
        """
        regime = np.full(num_laps, _GREEN, dtype=np.int8)
        p = self.params
        lap = 0
        while lap < num_laps:
            r = rng.random()
            if r < p.sc_prob_per_lap:
                dur = max(1, int(round(rng.normal(p.avg_sc_duration_laps,
                                                  p.avg_sc_duration_laps * 0.3))))
                regime[lap:lap + dur] = _SC
                lap += dur
            elif r < p.sc_prob_per_lap + p.vsc_prob_per_lap:
                dur = max(1, int(round(rng.normal(p.avg_vsc_duration_laps,
                                                  p.avg_vsc_duration_laps * 0.4))))
                regime[lap:lap + dur] = _VSC
                lap += dur
            else:
                lap += 1
        return regime[:num_laps]

    def _regime_factor(self, regime: np.ndarray) -> np.ndarray:
        """Map a regime array to per-lap time-inflation factors."""
        factor = np.ones_like(regime, dtype=float)
        factor[regime == _VSC] = self.params.vsc_lap_factor
        factor[regime == _SC]  = self.params.sc_lap_factor
        return factor

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        strategies: list[RaceResult],
        deg_uncertainty: dict[str, float] | None = None,
    ) -> list[StrategyDistribution]:
        """
        Run the Monte Carlo over the given deterministic strategy results.

        For each sampled scenario, every strategy's per-lap times are
        inflated by the active regime and its pit stops are discounted if
        they fall under a Safety Car (a stop under SC is much cheaper because
        the whole field is slow). All strategies see the same scenarios.

        ``deg_uncertainty`` (compound name → 1σ of degradation slope, from the
        learned tyre model) adds race-to-race degradation variability: each
        sampled race draws a per-compound slope offset that scales with tyre
        age, so "the tyres fell off a bit more than expected today" is modelled
        — the same draw is shared across strategies (common random numbers).
        """
        if not strategies:
            return []

        rng = np.random.default_rng(self.seed)
        n = self.num_samples
        num_laps = strategies[0].num_laps

        # Pre-extract per-lap arrays (raw physics+deg time, and pit loss).
        raw   = [np.array([lr.raw_lap_time for lr in s.laps]) for s in strategies]
        pit   = [np.array([lr.pit_time_loss for lr in s.laps]) for s in strategies]
        ns    = len(strategies)

        # Per-lap tyre age and compound (for the degradation-noise term).
        ages  = [_tyre_age_per_lap(s) for s in strategies]
        comps = [[lr.compound for lr in s.laps] for s in strategies]
        deg_unc = deg_uncertainty or {}
        deg_compounds = sorted(deg_unc.keys())

        totals   = np.zeros((n, ns))
        saw_sc   = np.zeros(n, dtype=bool)
        saw_any  = np.zeros(n, dtype=bool)

        for i in range(n):
            regime = self._sample_regime(num_laps, rng)
            factor = self._regime_factor(regime)
            saw_sc[i]  = bool(np.any(regime == _SC))
            saw_any[i] = bool(np.any(regime != _GREEN))
            noise = rng.normal(0.0, self.params.pace_noise_s, num_laps)
            under_event = factor > 1.0

            # One degradation-slope offset per compound for this race (shared
            # across strategies → common random numbers).
            deg_draw = {c: rng.normal(0.0, deg_unc[c]) for c in deg_compounds}

            for j in range(ns):
                rj = raw[j]
                pj = pit[j]
                # A pit stop under SC/VSC costs less: the field is slow, so the
                # pit-lane differential compresses by roughly the regime factor.
                pit_adj = np.where(under_event & (pj > 0.0), pj / factor, pj)
                lap_times = rj * factor + noise + pit_adj
                # Degradation noise: slope offset × tyre age, per compound.
                if deg_draw:
                    dterm = np.array([
                        deg_draw.get(comps[j][k], 0.0) * ages[j][k]
                        for k in range(len(rj))
                    ])
                    lap_times = lap_times + dterm
                totals[i, j] = lap_times.sum()

        # Win = fastest strategy in each scenario (paired comparison).
        winners = np.argmin(totals, axis=1)

        results: list[StrategyDistribution] = []
        for j, s in enumerate(strategies):
            col = totals[:, j]
            results.append(StrategyDistribution(
                name=s.strategy,
                deterministic_total=s.total_time,
                mean=float(np.mean(col)),
                p5=float(np.percentile(col, 5)),
                p50=float(np.percentile(col, 50)),
                p95=float(np.percentile(col, 95)),
                std=float(np.std(col)),
                win_probability=float(np.mean(winners == j)),
                sc_exposure_pct=float(np.mean(saw_sc) * 100.0),
                neutralisation_pct=float(np.mean(saw_any) * 100.0),
                samples=col.copy(),
            ))
        return results
