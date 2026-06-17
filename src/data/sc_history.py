# =========================================================
# Safety-Car history estimator
#
# Replaces the hand-written sc/vsc probabilities in the circuit YAMLs
# with estimates from REAL race-control data (FastF1 TrackStatus).
#
# Method
# ------
# For each historical race at the circuit we read the per-lap track
# status and count distinct SC / VSC deployments and their durations.
# The per-lap start probability is then estimated with EMPIRICAL BAYES
# shrinkage towards the global F1 average:
#
#     p = (deployments + p0 * n0) / (laps_observed + n0)
#
# where p0 is the global prior rate and n0 the prior strength in
# pseudo-laps. With only 5-6 races per circuit a raw frequency would be
# extremely noisy (one crash-heavy year doubles it); shrinkage keeps the
# estimate stable while still letting circuit character (street circuit
# vs open parkland) pull it away from the prior.
#
# FastF1 TrackStatus codes (per lap, possibly concatenated):
#   1 = green, 2 = yellow, 4 = SC deployed, 5 = red flag,
#   6 = VSC deployed, 7 = VSC ending
# A red flag is counted as an SC-like neutralisation (field slowed /
# stopped); for strategy purposes both give a cheap pit window.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path

_CACHE_DIR = Path("data/cache/sc_history")

# Global-F1 priors (per green lap) and prior strength in pseudo-laps.
# ~0.7 SC and ~0.6 VSC per race over a ~55-lap race, order of magnitude
# from recent seasons across all circuits.
_PRIOR_SC_PER_LAP  = 0.012
_PRIOR_VSC_PER_LAP = 0.010
_PRIOR_N0          = 150.0

# Duration priors (laps) and strength in pseudo-deployments. With only a
# handful of deployments per circuit the raw mean duration is very noisy
# (one long red-flag-style SC dominates), so it is shrunk towards a typical
# global duration just like the probability is.
_PRIOR_SC_DURATION  = 4.0
_PRIOR_VSC_DURATION = 2.5
_PRIOR_DUR_N0       = 3.0


def _shrink_duration(lengths: list[int], prior_dur: float) -> float:
    """Empirical-Bayes shrinkage of a mean duration towards a global prior."""
    n = len(lengths)
    if n == 0:
        return prior_dur
    raw_mean = sum(lengths) / n
    return (n * raw_mean + _PRIOR_DUR_N0 * prior_dur) / (n + _PRIOR_DUR_N0)


@dataclass(frozen=True)
class SCEstimate:
    """Historical Safety-Car statistics for one circuit."""
    circuit: str
    years_used: tuple[int, ...]
    races_used: int
    total_laps: int

    sc_deployments: int
    vsc_deployments: int
    sc_laps: int
    vsc_laps: int

    sc_prob_per_lap: float          # shrunk estimate
    vsc_prob_per_lap: float         # shrunk estimate
    avg_sc_duration_laps: float     # shrunk estimate
    avg_vsc_duration_laps: float    # shrunk estimate

    raw_sc_prob_per_lap: float      # un-shrunk frequency (for transparency)
    raw_vsc_prob_per_lap: float
    raw_avg_sc_duration_laps: float
    raw_avg_vsc_duration_laps: float


# ------------------------------------------------------------------ #
# Per-race extraction                                                  #
# ------------------------------------------------------------------ #

def _per_lap_status(fastf1_name: str, year: int) -> list[str] | None:
    """
    Return one combined TrackStatus string per race lap, or None if the
    session cannot be loaded (race not run, data missing, no network).
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache("data/fastf1_cache")
        session = fastf1.get_session(year, fastf1_name, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps is None or len(laps) == 0:
            return None
        grouped = laps.groupby("LapNumber")["TrackStatus"].apply(
            lambda x: "".join(sorted(set("".join(x.dropna().astype(str)))))
        )
        return [str(v) for _, v in sorted(grouped.items())]
    except Exception:
        return None


def _count_blocks(flags: list[bool]) -> tuple[int, list[int]]:
    """Count contiguous True-blocks and their lengths."""
    blocks, lengths = 0, []
    run = 0
    for f in flags:
        if f:
            run += 1
        elif run:
            blocks += 1
            lengths.append(run)
            run = 0
    if run:
        blocks += 1
        lengths.append(run)
    return blocks, lengths


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def estimate_sc_params(
    fastf1_name: str,
    years: list[int],
    use_cache: bool = True,
) -> SCEstimate | None:
    """
    Estimate SC/VSC per-lap probabilities for a circuit from history.

    Returns None when no historical race could be loaded (e.g. offline
    with a cold cache) — callers should fall back to the YAML values.
    """
    cache_file = _CACHE_DIR / f"{fastf1_name.lower().replace(' ', '_')}_{years[0]}_{years[-1]}.json"
    if use_cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        d["years_used"] = tuple(d["years_used"])
        return SCEstimate(**d)

    total_laps   = 0
    sc_dep, vsc_dep = 0, 0
    sc_laps, vsc_laps = 0, 0
    sc_lengths: list[int] = []
    vsc_lengths: list[int] = []
    years_ok: list[int] = []

    for year in years:
        statuses = _per_lap_status(fastf1_name, year)
        if statuses is None:
            continue
        years_ok.append(year)
        total_laps += len(statuses)

        # SC-like: SC deployed or red flag. VSC: codes 6/7 (and not SC).
        sc_flags  = [("4" in s) or ("5" in s) for s in statuses]
        vsc_flags = [(("6" in s) or ("7" in s)) and not f
                     for s, f in zip(statuses, sc_flags)]

        b, ln = _count_blocks(sc_flags)
        sc_dep += b; sc_lengths += ln; sc_laps += sum(ln)
        b, ln = _count_blocks(vsc_flags)
        vsc_dep += b; vsc_lengths += ln; vsc_laps += sum(ln)

    if not years_ok:
        return None

    green_laps = max(1, total_laps - sc_laps - vsc_laps)

    # Empirical-Bayes shrinkage towards the global prior.
    sc_p  = (sc_dep  + _PRIOR_SC_PER_LAP  * _PRIOR_N0) / (green_laps + _PRIOR_N0)
    vsc_p = (vsc_dep + _PRIOR_VSC_PER_LAP * _PRIOR_N0) / (green_laps + _PRIOR_N0)

    est = SCEstimate(
        circuit=fastf1_name,
        years_used=tuple(years_ok),
        races_used=len(years_ok),
        total_laps=total_laps,
        sc_deployments=sc_dep,
        vsc_deployments=vsc_dep,
        sc_laps=sc_laps,
        vsc_laps=vsc_laps,
        sc_prob_per_lap=round(sc_p, 5),
        vsc_prob_per_lap=round(vsc_p, 5),
        avg_sc_duration_laps=round(_shrink_duration(sc_lengths, _PRIOR_SC_DURATION), 2),
        avg_vsc_duration_laps=round(_shrink_duration(vsc_lengths, _PRIOR_VSC_DURATION), 2),
        raw_sc_prob_per_lap=round(sc_dep / green_laps, 5),
        raw_vsc_prob_per_lap=round(vsc_dep / green_laps, 5),
        raw_avg_sc_duration_laps=round(sum(sc_lengths) / len(sc_lengths), 2) if sc_lengths else 0.0,
        raw_avg_vsc_duration_laps=round(sum(vsc_lengths) / len(vsc_lengths), 2) if vsc_lengths else 0.0,
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(est), indent=2))
    return est
