# =========================================================
# Weather model
#
# WeatherModel — track wetness as a function of race lap.
#
# Level A (static)  : a single constant wetness for the whole race.
# Level B (dynamic) : a piecewise-linear timeline of (lap, wetness)
#                     keyframes, interpolated per lap, so the track can
#                     dry out or get wetter during the race (rain arriving,
#                     a drying line, etc.).
# Level C (forecast): an UNCERTAIN forecast — the rain onset lap, peak
#                     intensity and duration are random. Sampling it yields
#                     a distribution of Level-B timelines, which the weather
#                     Monte Carlo uses to score strategies on robustness to
#                     forecast uncertainty (not just one known timeline).
# =========================================================

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeatherModel:
    """
    Track wetness [0 = dry, 1 = soaked] over the course of a race.

    Parameters
    ----------
    keyframes : list[tuple[int, float]]
        Sorted (lap, wetness) control points. Wetness between keyframes is
        linearly interpolated; before the first / after the last keyframe it
        is held flat (clamped). A single keyframe = constant wetness.
    """

    keyframes: tuple[tuple[int, float], ...]

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def constant(cls, wetness: float) -> "WeatherModel":
        """A flat, time-invariant wetness (Level A static model)."""
        w = max(0.0, min(1.0, float(wetness)))
        return cls(keyframes=((1, w),))

    @classmethod
    def from_keyframes(cls, points: list[dict] | list[tuple[int, float]]) -> "WeatherModel":
        """
        Build from a list of {lap, wetness} dicts (YAML) or (lap, wetness)
        tuples. Points are sorted by lap and wetness is clamped to [0, 1].
        """
        kf: list[tuple[int, float]] = []
        for p in points:
            if isinstance(p, dict):
                lap, wet = int(p["lap"]), float(p["wetness"])
            else:
                lap, wet = int(p[0]), float(p[1])
            kf.append((lap, max(0.0, min(1.0, wet))))
        if not kf:
            kf = [(1, 0.0)]
        kf.sort(key=lambda x: x[0])
        return cls(keyframes=tuple(kf))

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def wetness(self, lap: int) -> float:
        """Interpolated track wetness at the given race lap (1-based)."""
        kf = self.keyframes
        if lap <= kf[0][0]:
            return kf[0][1]
        if lap >= kf[-1][0]:
            return kf[-1][1]
        for i in range(len(kf) - 1):
            l0, w0 = kf[i]
            l1, w1 = kf[i + 1]
            if l0 <= lap <= l1:
                if l1 == l0:
                    return w1
                frac = (lap - l0) / (l1 - l0)
                return w0 + frac * (w1 - w0)
        return kf[-1][1]

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def max_wetness(self) -> float:
        """Peak wetness over the whole timeline (drives compound availability)."""
        return max(w for _, w in self.keyframes)

    @property
    def is_dynamic(self) -> bool:
        """True if wetness changes during the race (Level B), else static."""
        ws = {round(w, 3) for _, w in self.keyframes}
        return len(ws) > 1

    def summary(self) -> str:
        """Human-readable one-line description for logs."""
        if not self.is_dynamic:
            w = self.keyframes[0][1]
            if w == 0.0:
                return "Dry"
            cond = "damp" if w < 0.55 else "wet" if w < 0.85 else "soaked"
            return f"WET (static) — wetness {w:.2f} ({cond})"
        pts = ", ".join(f"L{l}:{w:.2f}" for l, w in self.keyframes)
        return f"WET (dynamic) — peak {self.max_wetness:.2f}  [{pts}]"


@dataclass(frozen=True)
class WeatherForecast:
    """
    An UNCERTAIN rain forecast (Level C).

    Models a single rain shower whose timing and intensity are not known
    in advance — the situation a strategist actually faces ("rain expected
    around lap 25, maybe 60–80 % chance, could be heavy"). Sampling produces
    a concrete :class:`WeatherModel` timeline; the weather Monte Carlo samples
    many of these to score strategies on robustness to the forecast itself.

    Parameters
    ----------
    rain_probability : float
        P(it rains at all during the race). With probability ``1 −`` this,
        the sampled race stays dry.
    onset_lap_mean, onset_lap_std : float
        Lap at which the rain starts (Gaussian).
    peak_wetness_mean, peak_wetness_std : float
        Peak track wetness reached (Gaussian, clamped to [0, 1]).
    ramp_laps : int
        Laps taken to rise from dry to the peak (and to fall back).
    duration_laps_mean, duration_laps_std : float
        How long the wetness stays near its peak before drying out.
    race_laps : int
        Total race distance, used to clamp the timeline.
    """

    rain_probability: float
    onset_lap_mean: float
    onset_lap_std: float
    peak_wetness_mean: float
    peak_wetness_std: float
    ramp_laps: int
    duration_laps_mean: float
    duration_laps_std: float
    race_laps: int

    def sample(self, rng: np.random.Generator) -> WeatherModel:
        """Draw one concrete weather timeline from the forecast."""
        if rng.random() >= self.rain_probability:
            return WeatherModel.constant(0.0)

        onset = int(round(rng.normal(self.onset_lap_mean, self.onset_lap_std)))
        onset = max(1, min(self.race_laps, onset))
        peak = float(np.clip(
            rng.normal(self.peak_wetness_mean, self.peak_wetness_std), 0.05, 1.0))
        duration = max(1, int(round(
            rng.normal(self.duration_laps_mean, self.duration_laps_std))))
        ramp = max(1, self.ramp_laps)

        # Build a trapezoidal shower: dry → ramp up → plateau → ramp down → dry.
        start_dry = max(1, onset - 1)
        up        = min(self.race_laps, onset + ramp)
        plateau   = min(self.race_laps, up + duration)
        down      = min(self.race_laps, plateau + ramp)
        kf = [(start_dry, 0.0), (up, peak), (plateau, peak), (down, 0.0)]
        # Deduplicate laps (clamping can collide) keeping the wettest.
        merged: dict[int, float] = {}
        for lap, wet in kf:
            merged[lap] = max(merged.get(lap, 0.0), wet)
        return WeatherModel.from_keyframes(sorted(merged.items()))

    def summary(self) -> str:
        """Human-readable one-line description for logs."""
        return (
            f"FORECAST (uncertain) — P(rain) {self.rain_probability:.0%}, "
            f"onset L{self.onset_lap_mean:.0f}±{self.onset_lap_std:.0f}, "
            f"peak {self.peak_wetness_mean:.2f}±{self.peak_wetness_std:.2f}, "
            f"~{self.duration_laps_mean:.0f} laps"
        )
