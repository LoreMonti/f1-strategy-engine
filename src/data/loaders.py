"""
src/data/loaders.py
-------------------
Loader for real-circuit YAML files → Track / TyreCompound domain objects.

Adapated to the actual model signatures in:
  - src.models.track  : Track(name, segments, country)
                        TrackSegment(name, length, curvature, grip_factor,
                                     banking_deg, traction_factor, braking_severity)
  - src.models.tyre   : TyreCompound (frozen dataclass; pre-built SOFT/MEDIUM/HARD)
                        TYRE_COMPOUNDS dict

Design
------
The YAML carries a 20-point mini-sector speed/throttle trace plus 3 official
sectors.  We convert both into TrackSegment objects so the existing integrator
can consume them directly without any changes.

Two mapping strategies are available (set via ``segment_source`` kwarg):

  "mini_sectors"  (default)
      20 segments derived from the speed trace.  Each segment gets a
      curvature estimate from lateral_g, and braking_severity from the
      brake channel.  Best fidelity for the physics integrator.

  "sectors"
      3 coarse segments (one per official F1 sector).  Lighter-weight;
      useful for unit tests or quick smoke-tests.

Usage
-----
    from src.data.loaders import TrackLoader

    loader = TrackLoader("data/tracks/monza_2024.yaml")
    track  = loader.track()                       # → Track
    tyres  = loader.tyre_compounds()              # → dict[str, TyreCompound]
    info   = loader.race_info()                   # → RaceInfo
    env    = loader.environment()                 # → EnvironmentConditions

    # Coarse 3-segment version:
    track3 = loader.track(segment_source="sectors")

CLI smoke-test
--------------
    python -m src.data.loaders data/tracks/monza_2024.yaml
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.models.track import Track, TrackSegment
from src.models.tyre import TYRE_COMPOUNDS, TyreCompound

# Vehicle parameter keys that circuit YAMLs are allowed to override
_VEHICLE_OVERRIDE_KEYS = frozenset({
    "drag_coefficient", "lift_coefficient", "max_speed",
    "fuel_mass", "tyre_mu", "max_brake_accel", "ers_power_kw",
})


# ── Lightweight output dataclasses ──────────────────────────────────────────

@dataclass
class RaceInfo:
    circuit_id: str
    name: str
    gp_name: str
    country: str
    season: int
    race_laps: int
    lap_distance_km: float
    drs_zones: int
    pit_lane_delta_s: float
    stationary_time_s: float
    fuel_load_kg: float
    fuel_effect_s_per_kg: float
    fuel_consumption_kg_per_lap: float
    sc_probability_per_lap: float
    vsc_probability_per_lap: float
    avg_sc_duration_laps: int


@dataclass
class EnvironmentConditions:
    altitude_m: float
    ambient_temp_c: float
    track_temp_c: float
    humidity_pct: float
    wind_speed_ms: float
    grip_factor: float
    tyre_stress_factor: float
    brake_stress_factor: float
    bump_severity: float


# ── Exception ────────────────────────────────────────────────────────────────

class TrackLoadError(ValueError):
    """Raised when the YAML track file is invalid or incomplete."""


# ── Helpers ──────────────────────────────────────────────────────────────────

# Gravity constant for lateral-g → curvature conversion
_G = 9.81  # m/s²

# Maps YAML compound labels to keys in TYRE_COMPOUNDS
_COMPOUND_LABEL_MAP: dict[str, str] = {
    "Hard":         "hard",
    "Medium":       "medium",
    "Soft":         "soft",
    "Intermediate": "intermediate",
    "Wet":          "wet",
    # also accept lowercase directly
    "hard":         "hard",
    "medium":       "medium",
    "soft":         "soft",
    "intermediate": "intermediate",
    "wet":          "wet",
}


def _lateral_g_to_curvature(lateral_g: float, speed_kph: float) -> float:
    """
    Estimate track curvature [1/m] from lateral acceleration and speed.

    lateral_g   – peak lateral load in g units
    speed_kph   – representative corner speed [km/h]

    Derivation:  a_lat = v² * κ  →  κ = a_lat / v²
    """
    if speed_kph <= 0 or lateral_g <= 0:
        return 0.0
    v_ms = speed_kph / 3.6
    a_lat = lateral_g * _G
    return a_lat / (v_ms ** 2)


# ── Main loader ───────────────────────────────────────────────────────────────

class TrackLoader:
    """
    Parse a track YAML and expose typed accessors for domain objects.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the circuit YAML (e.g. ``data/tracks/monza_2024.yaml``).
    """

    _REQUIRED_TOP = {
        "metadata", "environment", "surface",
        "sectors", "mini_sectors", "tyres",
        "pit_stop", "fuel", "safety_car",
    }

    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Track YAML not found: {self._path}")

        with self._path.open("r", encoding="utf-8") as fh:
            self._raw: dict[str, Any] = yaml.safe_load(fh)

        self._validate()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self) -> None:
        missing = self._REQUIRED_TOP - set(self._raw.keys())
        if missing:
            raise TrackLoadError(
                f"{self._path.name}: missing top-level keys: {sorted(missing)}"
            )

        # Sector distances must sum within 50 m of declared lap distance
        total_m = sum(s["distance_m"] for s in self._raw["sectors"])
        declared_m = self._raw["metadata"]["lap_distance_km"] * 1000
        if abs(total_m - declared_m) > 50:
            raise TrackLoadError(
                f"Sector distances sum to {total_m:.0f} m but "
                f"lap_distance_km × 1000 = {declared_m:.0f} m "
                f"(delta {abs(total_m - declared_m):.0f} m > 50 m tolerance)"
            )

    # ── Plain-dataclass accessors (no domain model dependency) ───────────────

    def race_info(self) -> RaceInfo:
        m   = self._raw["metadata"]
        pit = self._raw["pit_stop"]
        f   = self._raw["fuel"]
        sc  = self._raw["safety_car"]
        return RaceInfo(
            circuit_id=m["circuit_id"],
            name=m["name"],
            gp_name=m["gp_name"],
            country=m["country"],
            season=m["season"],
            race_laps=m["race_laps"],
            lap_distance_km=m["lap_distance_km"],
            drs_zones=m["drs_zones"],
            pit_lane_delta_s=pit["pit_lane_delta_s"],
            stationary_time_s=pit["stationary_time_s"],
            fuel_load_kg=f["race_fuel_load_kg"],
            fuel_effect_s_per_kg=f["fuel_effect_s_per_kg"],
            fuel_consumption_kg_per_lap=f["consumption_kg_per_lap"],
            sc_probability_per_lap=sc["sc_probability_per_lap"],
            vsc_probability_per_lap=sc["vsc_probability_per_lap"],
            avg_sc_duration_laps=sc.get("avg_sc_duration_laps", 4),
        )

    def track_wetness(self) -> float:
        """
        Static track wetness [0 = dry, 1 = soaked] for the Level A weather model.

        Read from the optional ``weather: {track_wetness: X}`` YAML section.
        Defaults to 0.0 (dry) when the section is absent.
        """
        w = self._raw.get("weather", {}).get("track_wetness", 0.0)
        return max(0.0, min(1.0, float(w)))

    def weather_model(self) -> "WeatherModel":
        """
        Build the circuit's weather model.

        Reads the optional ``weather`` YAML section:
          - ``timeline: [{lap, wetness}, ...]`` → Level B dynamic evolution
          - else ``track_wetness: X``          → Level A static wetness
          - absent                             → dry.
        """
        from src.models.weather import WeatherModel
        weather = self._raw.get("weather", {})
        timeline = weather.get("timeline")
        if timeline:
            return WeatherModel.from_keyframes(timeline)
        return WeatherModel.constant(self.track_wetness())

    def environment(self) -> EnvironmentConditions:
        e = self._raw["environment"]
        s = self._raw["surface"]
        return EnvironmentConditions(
            altitude_m=e["altitude_m"],
            ambient_temp_c=e["ambient_temp_c"],
            track_temp_c=e["track_temp_c"],
            humidity_pct=e["humidity_pct"],
            wind_speed_ms=e["typical_wind_speed_ms"],
            grip_factor=s["grip_factor"],
            tyre_stress_factor=s["tyre_stress_factor"],
            brake_stress_factor=s["brake_stress_factor"],
            bump_severity=s["bump_severity"],
        )

    # ── Domain-model constructors ─────────────────────────────────────────────

    def track(
        self,
        segment_source: Literal["mini_sectors", "sectors"] = "mini_sectors",
    ) -> Track:
        """
        Build a ``Track`` object from the YAML.

        Parameters
        ----------
        segment_source : "mini_sectors" | "sectors"
            Which YAML data to convert into TrackSegment objects.
            - "mini_sectors": 20 segments with physics-grade fidelity (default).
            - "sectors": 3 coarse segments (one per official F1 sector).
        """
        ri  = self.race_info()
        env = self.environment()

        if segment_source == "mini_sectors":
            segments = self._segments_from_mini_sectors(env)
        elif segment_source == "sectors":
            segments = self._segments_from_sectors(env)
        else:
            raise ValueError(
                f"segment_source must be 'mini_sectors' or 'sectors', "
                f"got '{segment_source}'"
            )

        return Track(
            name=ri.name,
            segments=segments,
            country=ri.country,
        )

    def _segments_from_mini_sectors(
        self, env: EnvironmentConditions
    ) -> list[TrackSegment]:
        """
        Convert the 20 mini-sector rows into TrackSegment objects.

        Each mini-sector is split into up to three sub-segments so that the
        corner speed limit is only applied at the actual apex, not across the
        entire ~290 m zone.  This gives realistic lap times (≈ 84-88 s at Monza
        in race conditions) instead of the ~20 s penalty that results from
        treating the full braking zone as a constant-speed segment.

        Sub-segment breakdown for a corner mini-sector
        -----------------------------------------------
        approach  – braking in a straight line (curvature = 0)
        apex      – short segment with full corner curvature
        exit      – acceleration zone (curvature = 0)

        For pure straights (lateral_g < 0.8) a single segment is emitted.

        Mapping
        -------
        YAML field        → TrackSegment field
        ─────────────────────────────────────────────────────────────
        dist_m (delta)    → length          (m between consecutive rows)
        lateral_g+speed   → curvature       (1/m, apex sub-segment only)
        surface.grip      → grip_factor     (circuit-wide constant)
        brake             → braking_severity & approach length fraction
        throttle          → traction_factor
        banking not in YAML → banking_deg = 0.0 (Monza is flat)
        """
        rows = self._raw["mini_sectors"]
        lap_m = self._raw["metadata"]["lap_distance_km"] * 1000
        n = len(rows)
        segments: list[TrackSegment] = []

        # Minimum apex length must exceed STRATEGY_STEP_SIZE (50 m) so the
        # backward braking pass always sees at least one interior sample point
        # inside the corner.  Without this, coarse step sizes skip the apex
        # constraint entirely, producing lap times ~9 s too fast.
        _MIN_LEN = 55.0

        for i, row in enumerate(rows):
            # Total mini-sector length
            next_dist = rows[i + 1]["dist_m"] if i < n - 1 else lap_m
            length = next_dist - row["dist_m"]
            if length <= 0:
                length = lap_m / n

            lateral_g  = row["lateral_g"]
            speed_kph  = row["speed_kph"]
            brake      = row["brake"]
            throttle   = row["throttle"]
            grip_factor = env.grip_factor
            label = f"MS{row['id']:02d}"

            braking_sev  = 0.5 + 1.5 * brake
            traction_fac = 0.7 + 0.6 * throttle

            # ── Pure straight (no meaningful cornering load) ──────────────
            if lateral_g < 0.8:
                segments.append(TrackSegment(
                    name=label,
                    length=length,
                    curvature=0.0,
                    grip_factor=grip_factor,
                    banking_deg=0.0,
                    traction_factor=round(traction_fac, 3),
                    braking_severity=round(braking_sev, 3),
                ))
                continue

            # ── Corner mini-sector: split into approach / apex / exit ────
            # Curvature from lateral_g and apex speed (used only for apex)
            curvature = _lateral_g_to_curvature(
                lateral_g=lateral_g,
                speed_kph=speed_kph,
            )

            # Approach fraction: proportional to braking intensity
            #   heavy brake (0.9) → ~45 % of segment is braking straight
            #   light brake (0.1) → ~5 %
            approach_frac = brake * 0.50

            # Apex fraction: shorter for tighter / harder-braking corners
            #   ranges from ~8 % (heavy brake, tight) to ~22 % (sweeper)
            if brake > 0.6:
                apex_frac = 0.10
            elif lateral_g > 2.5:
                apex_frac = 0.12
            elif lateral_g > 1.5:
                apex_frac = 0.18
            else:
                apex_frac = 0.22

            approach_len = length * approach_frac
            apex_len     = max(_MIN_LEN, length * apex_frac)
            exit_len     = length - approach_len - apex_len

            # Guard: if geometry collapses, fall back to single segment
            if exit_len < _MIN_LEN:
                apex_len  = max(_MIN_LEN, length - approach_len - _MIN_LEN)
                exit_len  = length - approach_len - apex_len
            if apex_len < _MIN_LEN or exit_len < _MIN_LEN:
                segments.append(TrackSegment(
                    name=label,
                    length=length,
                    curvature=round(curvature, 5),
                    grip_factor=grip_factor,
                    banking_deg=0.0,
                    traction_factor=round(traction_fac, 3),
                    braking_severity=round(braking_sev, 3),
                ))
                continue

            # If the approach is too short to be its own segment, fold its
            # length into the exit so the total lap distance is conserved.
            # (Previously this length was silently dropped, making every
            # light-braking corner ~10-120 m short and the whole lap 2-3 %
            # shorter than the real circuit.)
            if approach_len < _MIN_LEN:
                exit_len    += approach_len
                approach_len = 0.0

            # Approach sub-segment (braking straight, no curvature)
            if approach_len >= _MIN_LEN:
                segments.append(TrackSegment(
                    name=f"{label}a",
                    length=round(approach_len, 2),
                    curvature=0.0,
                    grip_factor=grip_factor,
                    banking_deg=0.0,
                    traction_factor=0.70,
                    braking_severity=round(braking_sev, 3),
                ))

            # Apex sub-segment (full curvature, speed-limited)
            segments.append(TrackSegment(
                name=f"{label}b",
                length=round(apex_len, 2),
                curvature=round(curvature, 5),
                grip_factor=grip_factor,
                banking_deg=0.0,
                traction_factor=0.78,
                braking_severity=1.0,
            ))

            # Exit sub-segment (acceleration, no curvature)
            segments.append(TrackSegment(
                name=f"{label}c",
                length=round(exit_len, 2),
                curvature=0.0,
                grip_factor=grip_factor,
                banking_deg=0.0,
                traction_factor=round(traction_fac, 3),
                braking_severity=0.55,
            ))

        return segments

    def _segments_from_sectors(
        self, env: EnvironmentConditions
    ) -> list[TrackSegment]:
        """
        Convert the 3 official F1 sectors into TrackSegment objects.

        Curvature is estimated from the sector average speed and a rough
        lateral_g proxy derived from min_speed_kph (tightest corner).
        """
        segments: list[TrackSegment] = []

        for raw in self._raw["sectors"]:
            # Rough curvature: use min_speed as the apex speed
            # Typical F1 lateral g at apex ~3.0–4.0 g; use 3.2 as default
            apex_g = 3.2
            curvature = _lateral_g_to_curvature(
                lateral_g=apex_g,
                speed_kph=raw["min_speed_kph"],
            )
            # Straights have curvature = 0
            if raw["type"] == "high_speed" and raw["min_speed_kph"] > 200:
                curvature = 0.0

            # braking_severity from brake_stress_factor in surface block
            braking_severity = env.brake_stress_factor

            segments.append(TrackSegment(
                name=f"S{raw['id']} {raw['name']}",
                length=float(raw["distance_m"]),
                curvature=round(curvature, 5),
                grip_factor=env.grip_factor,
                banking_deg=0.0,
                traction_factor=1.0,
                braking_severity=round(braking_severity, 3),
            ))

        return segments

    def tyre_compounds(self) -> dict[str, TyreCompound]:
        """
        Return TyreCompound objects for this event's nominated compounds.

        Circuit-specific overrides from the ``compound_overrides`` YAML section
        are applied on top of the base definitions in tyre.py using
        ``dataclasses.replace``.  Only fields that actually exist on
        TyreCompound are applied; unknown keys are silently ignored.

        Returns
        -------
        dict mapping compound label ("Hard", "Medium", "Soft") → TyreCompound
        """
        t         = self._raw["tyres"]
        overrides = self._raw.get("compound_overrides", {})
        result: dict[str, TyreCompound] = {}

        for code in t["available_compounds"]:
            label = t["compound_labels"][code]
            key   = _COMPOUND_LABEL_MAP.get(label)
            if key is None:
                raise TrackLoadError(
                    f"Unknown compound label '{label}' for code '{code}'. "
                    f"Expected one of {list(_COMPOUND_LABEL_MAP.keys())}"
                )
            if key not in TYRE_COMPOUNDS:
                raise TrackLoadError(
                    f"Compound key '{key}' not found in TYRE_COMPOUNDS. "
                    f"Available: {list(TYRE_COMPOUNDS.keys())}"
                )

            compound = TYRE_COMPOUNDS[key]

            if label in overrides:
                valid_fields = {
                    k: v for k, v in overrides[label].items()
                    if k in {f.name for f in dataclasses.fields(compound)}
                }
                if valid_fields:
                    compound = dataclasses.replace(compound, **valid_fields)

            result[label] = compound

        return result

    def fastf1_name(self) -> str | None:
        """Return the FastF1 circuit name, or None if not specified in YAML."""
        return self._raw["metadata"].get("fastf1_name")

    def mini_sectors_raw(self) -> list[dict]:
        """Return raw mini_sector rows (including `turn` field for track reconstruction)."""
        return self._raw["mini_sectors"]

    def vehicle_setup(self) -> dict:
        """
        Return circuit-specific vehicle parameter overrides.

        Only keys listed in ``_VEHICLE_OVERRIDE_KEYS`` are returned so that
        the caller cannot accidentally break unrelated Vehicle fields.
        Returns an empty dict if no ``vehicle_setup`` section is present.
        """
        vs = self._raw.get("vehicle_setup", {})
        return {k: v for k, v in vs.items() if k in _VEHICLE_OVERRIDE_KEYS}


# ── Convenience factory ───────────────────────────────────────────────────────

def load_track(yaml_path: str | Path) -> TrackLoader:
    """Shorthand:  ``loader = load_track("data/tracks/monza_2024.yaml")``"""
    return TrackLoader(yaml_path)


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/tracks/monza_2024.yaml"
    loader = TrackLoader(path)

    ri  = loader.race_info()
    env = loader.environment()

    print(f"\n{'─'*54}")
    print(f"  {ri.name}")
    print(f"  {ri.gp_name} {ri.season}  ·  {ri.race_laps} laps  ·  "
          f"{ri.lap_distance_km} km/lap")
    print(f"{'─'*54}")
    print(f"  Track temp  : {env.track_temp_c} °C   "
          f"Grip factor: {env.grip_factor}")
    print(f"  Pit delta   : {ri.pit_lane_delta_s} s   "
          f"Fuel load  : {ri.fuel_load_kg} kg")

    track_ms = loader.track(segment_source="mini_sectors")
    track_s3 = loader.track(segment_source="sectors")

    print(f"\n  Track (mini_sectors): {track_ms}")
    print(f"  Track (sectors)     : {track_s3}")

    print(f"\n  Segments (mini_sectors):")
    for seg in track_ms.segments:
        corner = f"κ={seg.curvature:.4f}" if not seg.is_straight else "straight"
        print(f"    {seg.name:<6s}  {seg.length:6.1f} m  "
              f"{corner:<18s}  "
              f"brake={seg.braking_severity:.2f}  "
              f"traction={seg.traction_factor:.2f}")

    print(f"\n  Tyres:")
    for label, tc in loader.tyre_compounds().items():
        print(f"    [{label:<6s}]  base_grip={tc.base_grip:.3f}  "
              f"wear/km={tc.wear_rate_per_km:.4f}  "
              f"opt_temp={tc.optimal_temperature:.0f} °C")
    print()
