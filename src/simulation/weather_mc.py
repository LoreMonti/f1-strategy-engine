# =========================================================
# Weather Monte Carlo (Level C — forecast uncertainty)
#
# The Level-B model runs ONE known weather timeline. Real strategy calls
# are made before the rain arrives, under an UNCERTAIN forecast: when it
# starts, how heavy, how long. This layer samples many timelines from a
# WeatherForecast and scores each candidate strategy across all of them,
# answering "which call is robust to the forecast being wrong?".
#
# Performance: the physics is NOT re-run inside the loop. A small per-
# compound lap-time response surface (lap time vs wetness) is precomputed
# once; each sampled race is then pure array math, exactly like the
# Safety-Car Monte Carlo. The weather term is applied as a DELTA from the
# nominal timeline, so the model is exact at the nominal forecast and only
# shifts as the sampled wetness departs from it.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.models.strategy import RaceResult
from src.models.weather import WeatherModel, WeatherForecast
from src.simulation.lap_simulator import LapSimulator


# Wetness grid for the response surface (interpolated between).
_BUCKETS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def build_wet_response(
    track,
    vehicle,
    compounds: dict,
    buckets: np.ndarray = _BUCKETS,
    step_size: float = 50.0,
    fuel_mass: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Precompute a lap-time response surface: ``surface[compound][bucket]`` is
    the lap time for that compound at that track wetness, at a fixed reference
    fuel load and a fresh tyre.

    Because the weather Monte Carlo uses this surface only as a *delta* from
    the nominal timeline, the absolute reference fuel cancels to first order;
    what matters is how each compound's lap time responds to wetness.
    """
    ref_fuel = fuel_mass if fuel_mass is not None else vehicle.fuel_mass * 0.5
    surface: dict[str, np.ndarray] = {}
    for name, comp in compounds.items():
        times = np.empty(len(buckets))
        for i, w in enumerate(buckets):
            sim = LapSimulator(track, vehicle, track_wetness=float(w))
            res = sim.simulate(
                step_size=step_size,
                tyre_compound=comp,
                initial_fuel_mass=ref_fuel,
            )
            times[i] = res["total_time"]
        surface[name] = times
    return surface


@dataclass
class WeatherStrategyDistribution:
    """Weather Monte Carlo outcome distribution for one strategy."""
    name: str
    deterministic_total: float      # race time under the nominal forecast
    mean: float
    p5: float
    p50: float
    p95: float
    std: float
    win_probability: float          # fraction of forecast scenarios won
    rain_exposure_pct: float        # % of sampled races that actually saw rain
    samples: np.ndarray = field(repr=False, default=None)

    @property
    def robustness_s(self) -> float:
        """Spread P95−P5: smaller = more robust to forecast uncertainty."""
        return self.p95 - self.p5


class WeatherMonteCarlo:
    """
    Score strategies under an uncertain rain forecast (Level C).

    Uses *common random numbers*: every strategy is evaluated against the
    SAME set of sampled weather timelines, so the win-probability comparison
    is a fair paired test.
    """

    def __init__(
        self,
        forecast: WeatherForecast,
        surface: dict[str, np.ndarray],
        nominal_weather: WeatherModel,
        buckets: np.ndarray = _BUCKETS,
        num_samples: int = 2000,
        seed: int | None = 42,
    ) -> None:
        self.forecast = forecast
        self.surface = surface
        self.nominal_weather = nominal_weather
        self.buckets = buckets
        self.num_samples = num_samples
        self.seed = seed

    def _wet_delta(self, compound: str, w: float, w_nom: float) -> float:
        """Lap-time change from moving wetness w_nom → w for this compound."""
        surf = self.surface.get(compound)
        if surf is None:
            return 0.0
        return float(np.interp(w, self.buckets, surf)
                     - np.interp(w_nom, self.buckets, surf))

    def evaluate(
        self, strategies: list[RaceResult]
    ) -> list[WeatherStrategyDistribution]:
        if not strategies:
            return []

        rng = np.random.default_rng(self.seed)
        n = self.num_samples
        num_laps = strategies[0].num_laps

        raw   = [np.array([lr.raw_lap_time for lr in s.laps]) for s in strategies]
        pit   = [np.array([lr.pit_time_loss for lr in s.laps]) for s in strategies]
        comps = [[lr.compound for lr in s.laps] for s in strategies]
        ns    = len(strategies)

        # Nominal per-lap wetness (the timeline the strategies were built on).
        w_nom = np.array([self.nominal_weather.wetness(k + 1)
                          for k in range(num_laps)])

        totals  = np.zeros((n, ns))
        saw_rain = np.zeros(n, dtype=bool)

        for i in range(n):
            wm = self.forecast.sample(rng)
            w_lap = np.array([wm.wetness(k + 1) for k in range(num_laps)])
            saw_rain[i] = bool(np.any(w_lap > 0.05))

            for j in range(ns):
                delta = np.array([
                    self._wet_delta(comps[j][k], w_lap[k], w_nom[k])
                    for k in range(num_laps)
                ])
                lap_times = raw[j] + delta + pit[j]
                totals[i, j] = lap_times.sum()

        winners = np.argmin(totals, axis=1)
        results: list[WeatherStrategyDistribution] = []
        for j, s in enumerate(strategies):
            col = totals[:, j]
            results.append(WeatherStrategyDistribution(
                name=s.strategy,
                deterministic_total=s.total_time,
                mean=float(np.mean(col)),
                p5=float(np.percentile(col, 5)),
                p50=float(np.percentile(col, 50)),
                p95=float(np.percentile(col, 95)),
                std=float(np.std(col)),
                win_probability=float(np.mean(winners == j)),
                rain_exposure_pct=float(np.mean(saw_rain) * 100.0),
                samples=col.copy(),
            ))
        return results
