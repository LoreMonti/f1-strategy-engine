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
# =========================================================

from __future__ import annotations
from dataclasses import dataclass


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
