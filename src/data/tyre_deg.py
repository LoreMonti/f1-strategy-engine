# =========================================================
# Tyre degradation — learned from real race stints
#
# Replaces the hand-tuned `deg_s_per_lap` overlay with a value FITTED
# to real FastF1 stint data, per compound and per circuit, WITH an
# uncertainty estimate that feeds the Monte Carlo degradation noise.
#
# Why ML/stats here (and not just physics):
#   Tyre thermal/chemical degradation is the part of the model the
#   physics captures worst — it is genuinely data-driven in the paddock.
#   Teams fit it from stint telemetry, exactly as done here.
#
# Method (standard paddock stint analysis)
# ----------------------------------------
#   1. Pull every clean green-flag stint (FastF1 IsAccurate flag + green
#      TrackStatus), drop the out-lap and ~2 warm-up laps.
#   2. Fuel-correct each lap with the circuit's known fuel effect:
#         t_fc = t_lap + (fuel_s_per_lap) * lap_number
#      so the only remaining trend vs tyre age is degradation.
#   3. Robust slope per stint via Theil–Sen (resistant to traffic laps).
#   4. Aggregate slopes per (circuit, compound); shrink the per-circuit
#      estimate towards the compound's global mean (partial pooling) —
#      same empirical-Bayes idea as the Safety-Car estimator, needed
#      because a single circuit has only a handful of stints.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

_CACHE_DIR = Path("data/cache/tyre_deg")

# Partial-pooling strength (pseudo-stints) towards the global compound mean.
_PRIOR_N0 = 5.0

# Minimum clean laps and age span for a stint to be usable.
_MIN_STINT_LAPS = 6
_MIN_AGE_SPAN   = 5
_WARMUP_AGE     = 3   # ignore tyre_age < 3 (out-lap + warm-up)


@dataclass(frozen=True)
class CompoundDeg:
    """Learned degradation for one (circuit, compound)."""
    compound: str
    deg_s_per_lap: float        # shrunk robust estimate
    deg_std: float              # spread across stints → Monte Carlo noise
    n_stints: int
    raw_deg_s_per_lap: float    # un-shrunk circuit median (transparency)
    global_deg_s_per_lap: float # compound global mean (the prior)


# ------------------------------------------------------------------ #
# Extraction                                                           #
# ------------------------------------------------------------------ #

def extract_stints(fastf1_name: str, years: list[int]) -> list[dict]:
    """
    Return clean green-flag stints for a circuit across seasons.

    Each stint: {compound, driver, year, age[], lap[], time[]} with the
    out-lap and warm-up laps already removed.
    """
    try:
        import fastf1
        import pandas as pd
        fastf1.Cache.enable_cache("data/fastf1_cache")
    except Exception:
        return []

    stints: list[dict] = []
    for year in years:
        try:
            session = fastf1.get_session(year, fastf1_name, "R")
            session.load(telemetry=False, weather=False, messages=False)
            laps = session.laps
        except Exception:
            continue
        if laps is None or len(laps) == 0:
            continue

        for (drv, stint_id), grp in laps.groupby(["Driver", "Stint"]):
            grp = grp.sort_values("LapNumber")
            ages, lapnums, times, comp = [], [], [], None
            for _, r in grp.iterrows():
                lt = r.get("LapTime")
                if lt is None or pd.isna(lt):
                    continue
                if not bool(r.get("IsAccurate", False)):
                    continue
                if str(r.get("TrackStatus", "1")) != "1":   # green only
                    continue
                age = r.get("TyreLife")
                if age is None or pd.isna(age) or age < _WARMUP_AGE:
                    continue
                comp = str(r.get("Compound", "UNKNOWN")).capitalize()
                ages.append(float(age))
                lapnums.append(float(r["LapNumber"]))
                times.append(lt.total_seconds())

            if len(ages) < _MIN_STINT_LAPS:
                continue
            if max(ages) - min(ages) < _MIN_AGE_SPAN:
                continue
            if comp in (None, "Unknown", "Nan"):
                continue

            stints.append({
                "compound": comp, "driver": drv, "year": year,
                "age": np.array(ages), "lap": np.array(lapnums),
                "time": np.array(times),
            })
    return stints


# ------------------------------------------------------------------ #
# Fitting                                                              #
# ------------------------------------------------------------------ #

def _stint_slope(age: np.ndarray, time_fc: np.ndarray) -> float | None:
    """Robust degradation slope [s/lap] for one stint via Theil–Sen."""
    if len(age) < _MIN_STINT_LAPS:
        return None
    # Drop intra-stint outliers (traffic/lock-ups): keep within 1.5 s of median.
    med = np.median(time_fc)
    keep = np.abs(time_fc - med) < 1.5
    if keep.sum() < _MIN_STINT_LAPS:
        keep = np.ones_like(time_fc, dtype=bool)
    from scipy.stats import theilslopes
    slope, *_ = theilslopes(time_fc[keep], age[keep])
    return float(slope)


def learn_degradation(
    fastf1_name: str,
    years: list[int],
    fuel_s_per_lap: float,
    global_prior: dict[str, float] | None = None,
    use_cache: bool = True,
) -> dict[str, CompoundDeg]:
    """
    Learn per-compound degradation (s/lap) with uncertainty for a circuit.

    ``fuel_s_per_lap`` removes the fuel-burn trend before fitting.
    ``global_prior`` maps compound → global mean slope; when None the
    circuit's own pooled mean is used as the prior (no cross-circuit pull).
    """
    safe = fastf1_name.lower().replace(" ", "_")
    cache_file = _CACHE_DIR / f"{safe}_{years[0]}_{years[-1]}.json"
    if use_cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        return {k: CompoundDeg(**v) for k, v in d.items()}

    stints = extract_stints(fastf1_name, years)

    # Per-compound stint slopes (fuel-corrected), keyed also by (driver, year)
    # so we can separate systematic car/era variance from genuine race-to-race
    # uncertainty.
    slopes: dict[str, list[float]] = {}
    groups: dict[tuple, list[float]] = {}
    for st in stints:
        time_fc = st["time"] + fuel_s_per_lap * st["lap"]
        s = _stint_slope(st["age"], time_fc)
        if s is not None:
            slopes.setdefault(st["compound"], []).append(s)
            groups.setdefault((st["compound"], st["driver"], st["year"]), []).append(s)

    # Systematic-variance deflation: the cross-stint spread mixes different
    # cars/drivers/eras. The portion that is genuine per-race uncertainty is
    # the residual WITHIN a (driver, year) group. We pool those residuals
    # across compounds (few per group) to get one global deflation ratio.
    within_res, cross_res = [], []
    for comp in slopes:
        arr = np.array(slopes[comp])
        cross_res += list(arr - np.median(arr))
    for _, v in groups.items():
        if len(v) >= 2:
            v = np.array(v)
            within_res += list(v - np.median(v))
    if len(within_res) >= 8 and len(cross_res) >= 8:
        ratio = np.std(within_res) / max(np.std(cross_res), 1e-6)
        deflation = float(np.clip(ratio, 0.3, 1.0))
    else:
        deflation = 0.6   # fallback when within-group data is too thin

    result: dict[str, CompoundDeg] = {}
    for comp, sl in slopes.items():
        arr = np.array(sl)
        raw_med = float(np.median(arr))
        prior = (global_prior or {}).get(comp, raw_med)
        n = len(arr)
        shrunk = (n * raw_med + _PRIOR_N0 * prior) / (n + _PRIOR_N0)
        # Robust spread → std for the Monte Carlo (MAD scaled to σ), then
        # deflated to the genuine per-race uncertainty (systematic car/era
        # variance removed).
        mad = float(np.median(np.abs(arr - raw_med)))
        std = max(0.005, 1.4826 * mad * deflation)
        result[comp] = CompoundDeg(
            compound=comp,
            deg_s_per_lap=round(max(0.0, shrunk), 4),
            deg_std=round(std, 4),
            n_stints=n,
            raw_deg_s_per_lap=round(raw_med, 4),
            global_deg_s_per_lap=round(prior, 4),
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({k: asdict(v) for k, v in result.items()}, indent=2))
    return result
