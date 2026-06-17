# =========================================================
# FastF1 Loader
#
# Downloads real F1 data from FastF1 API and caches locally.
#
# Functions:
#   get_track_map(name, year)         → GPS track centerline
#   get_qualifying_telemetry(name, year) → fastest Q lap telemetry
#   get_race_laps(name, year)         → race lap times + compounds
# =========================================================

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_CACHE_DIR = Path("data/fastf1_cache")
_ARRAY_DIR = Path("data/cache/track_maps")


def get_track_map(
    fastf1_name: str,
    year: int,
    session_type: str = "Q",
    force_refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return GPS-derived track centerline for the given circuit.

    Uses the fastest qualifying lap telemetry X/Y channels.
    Results are cached in data/cache/track_maps/ as .npz files.

    Parameters
    ----------
    fastf1_name : circuit name as FastF1 recognises it (e.g. "Monza")
    year        : season year (e.g. 2024)
    session_type: FastF1 session identifier ("Q" = qualifying, "R" = race)
    force_refresh: if True, ignore cache and re-download

    Returns
    -------
    s_arr : cumulative distance along the track [m]
    x_arr : x coordinates [m]
    y_arr : y coordinates [m]
    """
    _ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _ARRAY_DIR / f"{fastf1_name.lower()}_{year}_{session_type}.npz"

    if cache_file.exists() and not force_refresh:
        data = np.load(cache_file)
        print(f"[FastF1] Track map loaded from cache: {cache_file.name}")
        return data["s"], data["x"], data["y"]

    print(f"[FastF1] Downloading track map: {fastf1_name} {year} {session_type} …")
    print("         (this may take a minute on first run)")

    import fastf1  # lazy import so the rest of the project doesn't require it

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(_CACHE_DIR))

    session = fastf1.get_session(year, fastf1_name, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry().add_distance()

    # X, Y are car positions in decimetres in FastF1's circuit coordinate system.
    # Divide by 10 to convert to metres.
    x_raw = tel["X"].values.astype(float) / 10.0
    y_raw = tel["Y"].values.astype(float) / 10.0
    s_raw = tel["Distance"].values.astype(float)

    # Remove NaNs
    mask = ~(np.isnan(x_raw) | np.isnan(y_raw) | np.isnan(s_raw))
    x_raw, y_raw, s_raw = x_raw[mask], y_raw[mask], s_raw[mask]

    # Normalise: translate so start is at origin, rotate so first segment → east
    x_raw -= x_raw[0]
    y_raw -= y_raw[0]

    # Rotate so the pit straight runs left → right (first ~500 m horizontal)
    n_ref = min(100, len(x_raw) - 1)
    dx0 = x_raw[n_ref] - x_raw[0]
    dy0 = y_raw[n_ref] - y_raw[0]
    rot = -np.arctan2(dy0, dx0)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    x_rot = x_raw * cos_r - y_raw * sin_r
    y_rot = x_raw * sin_r + y_raw * cos_r

    np.savez(cache_file, s=s_raw, x=x_rot, y=y_rot)
    print(f"[FastF1] Track map cached → {cache_file}")

    return s_raw, x_rot, y_rot


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enable_cache() -> None:
    import fastf1
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(_CACHE_DIR))


def _load_session(year: int, name: str, session_type: str):
    import fastf1
    _enable_cache()
    session = fastf1.get_session(year, name, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    return session


# ── Qualifying telemetry ──────────────────────────────────────────────────────

def get_qualifying_telemetry(
    fastf1_name: str,
    year: int,
    force_refresh: bool = False,
) -> dict:
    """
    Return fastest qualifying lap telemetry aligned to distance.

    Returns
    -------
    dict with keys:
        driver      : driver abbreviation (e.g. "LEC")
        lap_time_s  : lap time in seconds
        s           : distance array [m]
        speed_kmh   : speed [km/h]
        throttle    : throttle 0-1
        brake       : brake 0-1
        gear        : gear number
    """
    cache_dir = Path("data/cache/telemetry")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{fastf1_name.lower()}_{year}_Q_tel.npz"
    meta_file  = cache_dir / f"{fastf1_name.lower()}_{year}_Q_tel_meta.txt"

    if cache_file.exists() and meta_file.exists() and not force_refresh:
        data   = np.load(cache_file)
        meta   = meta_file.read_text().strip().split(",")
        driver, lap_time_s = meta[0], float(meta[1])
        print(f"[FastF1] Q telemetry loaded from cache ({driver}, {lap_time_s:.3f}s)")
        return dict(
            driver=driver, lap_time_s=lap_time_s,
            s=data["s"], speed_kmh=data["speed_kmh"],
            throttle=data["throttle"], brake=data["brake"],
            gear=data["gear"],
        )

    print(f"[FastF1] Downloading Q telemetry: {fastf1_name} {year} …")
    session = _load_session(year, fastf1_name, "Q")
    fastest = session.laps.pick_fastest()
    tel     = fastest.get_telemetry().add_distance()

    s         = tel["Distance"].values.astype(float)
    speed_kmh = tel["Speed"].values.astype(float)
    throttle  = tel["Throttle"].values.astype(float) / 100.0
    brake     = tel["Brake"].values.astype(float).clip(0, 1)
    gear      = tel["nGear"].values.astype(float)

    # Remove NaNs
    mask = ~(np.isnan(s) | np.isnan(speed_kmh))
    s, speed_kmh, throttle, brake, gear = (
        s[mask], speed_kmh[mask], throttle[mask], brake[mask], gear[mask]
    )

    driver     = fastest["Driver"]
    lap_time_s = fastest["LapTime"].total_seconds()

    np.savez(cache_file, s=s, speed_kmh=speed_kmh,
             throttle=throttle, brake=brake, gear=gear)
    meta_file.write_text(f"{driver},{lap_time_s:.6f}")
    print(f"[FastF1] Q telemetry cached ({driver}, {lap_time_s:.3f}s)")

    return dict(
        driver=driver, lap_time_s=lap_time_s,
        s=s, speed_kmh=speed_kmh,
        throttle=throttle, brake=brake, gear=gear,
    )


# ── Race lap times ────────────────────────────────────────────────────────────

def get_race_laps(
    fastf1_name: str,
    year: int,
    driver: str | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Return race lap times for a driver (default: race winner).

    Returns
    -------
    dict with keys:
        driver      : driver abbreviation
        laps        : list of dicts {lap, compound, lap_time_s, tyre_life, pit_in}
        fastest_lap_s : fastest clean lap time
        avg_lap_s   : average clean lap time
    """
    cache_dir = Path("data/cache/race_laps")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag        = driver.upper() if driver else "WIN"
    cache_file = cache_dir / f"{fastf1_name.lower()}_{year}_R_{tag}.npz"
    meta_file  = cache_dir / f"{fastf1_name.lower()}_{year}_R_{tag}_meta.txt"

    if cache_file.exists() and meta_file.exists() and not force_refresh:
        data        = np.load(cache_file, allow_pickle=True)
        meta        = meta_file.read_text().strip().split(",")
        drv         = meta[0]
        laps_raw    = data["laps"].tolist()
        print(f"[FastF1] Race laps loaded from cache ({drv}, {len(laps_raw)} laps)")
        return _race_laps_dict(drv, laps_raw)

    print(f"[FastF1] Downloading race laps: {fastf1_name} {year} …")
    session = _load_session(year, fastf1_name, "R")

    if driver:
        drv_laps = session.laps.pick_driver(driver.upper())
        drv = driver.upper()
    else:
        # Pick the race winner from the official classification (finishing
        # position == 1). This is robust; summing LapTime and taking idxmin()
        # is NOT — pandas skips NaT laps in the sum, so a retired driver with
        # many missing lap times gets an artificially small total and is
        # wrongly selected as the "winner" (e.g. SAR at Bahrain 2024).
        drv = None
        try:
            results = session.results
            pos = results["Position"]
            winner_row = results[pos == 1]
            if len(winner_row):
                drv = str(winner_row.iloc[0]["Abbreviation"])
            else:
                # results are sorted by finishing order → first row is the winner
                drv = str(results.iloc[0]["Abbreviation"])
        except Exception:
            drv = None

        if not drv:
            # Fallback: among finishers (≥90% laps completed), smallest total time.
            all_laps = session.laps
            lap_counts = all_laps.groupby("Driver")["LapNumber"].max().dropna()
            max_laps_in_race = int(lap_counts.max())
            finishers = lap_counts[lap_counts >= max_laps_in_race * 0.9].index
            race_times = (
                all_laps[all_laps["Driver"].isin(finishers)]
                .groupby("Driver")["LapTime"]
                .sum()
                .dropna()
            )
            drv = race_times.idxmin()

        drv_laps = session.laps.pick_driver(drv)

    laps_raw = []
    for _, row in drv_laps.iterrows():
        lt = row.get("LapTime")
        if lt is None or (hasattr(lt, "isnull") and lt.isnull()):
            continue
        try:
            lt_s = lt.total_seconds()
        except Exception:
            continue
        if lt_s <= 0 or lt_s > 300:   # filter formation lap / safety car outliers
            continue

        compound  = str(row.get("Compound", "UNKNOWN")).capitalize()
        tyre_life = int(row.get("TyreLife", 0) or 0)
        lap_num   = int(row.get("LapNumber", 0) or 0)
        import pandas as _pd
        _pit_val  = row.get("PitInTime")
        try:
            pit_in = bool(_pit_val is not None and not _pd.isna(_pit_val))
        except (TypeError, ValueError):
            pit_in = False

        laps_raw.append({
            "lap": lap_num,
            "compound": compound,
            "lap_time_s": lt_s,
            "tyre_life": tyre_life,
            "pit_in": pit_in,
        })

    laps_raw.sort(key=lambda x: x["lap"])

    np.savez(cache_file, laps=np.array(laps_raw, dtype=object))
    meta_file.write_text(f"{drv}")
    print(f"[FastF1] Race laps cached ({drv}, {len(laps_raw)} laps)")

    return _race_laps_dict(drv, laps_raw)


_WET_COMPOUNDS = {"Intermediate", "Wet"}

def _race_laps_dict(driver: str, laps: list[dict]) -> dict:
    all_times = [l["lap_time_s"] for l in laps]
    median_t = float(np.median(all_times)) if all_times else 90.0
    # Exclude pit-stop laps and SC/VSC outliers (>10% above median)
    clean = [
        l for l in laps
        if not l.get("pit_in", False) and l["lap_time_s"] <= median_t * 1.10
    ]
    times = [l["lap_time_s"] for l in clean]

    # Detect wet/mixed race: if >20% of laps used Intermediate or Wet compound
    wet_count = sum(1 for l in laps if l.get("compound", "") in _WET_COMPOUNDS)
    wet_race = wet_count / max(len(laps), 1) > 0.20

    return dict(
        driver=driver,
        laps=laps,
        median_lap_s=median_t,
        fastest_lap_s=min(times) if times else 0.0,
        avg_lap_s=float(np.mean(times)) if times else 0.0,
        wet_race=wet_race,
    )
