# =========================================================
# Tyre model
#
# TyreCompound  — immutable compound specification
# TyreState     — mutable per-lap tyre state
# =========================================================

from __future__ import annotations
from dataclasses import dataclass
import math

from src.utils.units import m_to_km


@dataclass(frozen=True)
class TyreCompound:
    """
    Immutable specification for a tyre compound.

    Grip / wear model
    -----------------
    base_grip : float
        Initial grip multiplier at zero wear and optimal temperature.
    wear_rate_per_km : float
        Wear increment per km of distance [wear/km].
    min_grip_multiplier : float
        Hard floor on grip — never goes below this regardless of wear.
    cliff_wear : float
        Wear level [0–1] at which the cliff effect starts.
    cliff_penalty : float
        Additional grip loss per unit of wear beyond the cliff.

    Temperature model
    -----------------
    optimal_temperature : float
        Temperature [°C] at which grip is maximised.
    cold_temperature : float
        Temperature [°C] below which grip starts to drop noticeably.
    overheating_temperature : float
        Temperature [°C] above which the tyre overheats.
    thermal_sensitivity : float
        Grip loss per °C away from optimal temperature.
    heating_rate : float
        Scaling factor for temperature gain under load.
    cooling_rate : float
        Scaling factor for temperature loss from airflow.

    Wear thermal sensitivity
    ------------------------
    cold_wear_multiplier : float
        Extra wear rate when below cold_temperature.
    hot_wear_multiplier : float
        Extra wear rate when above overheating_temperature.
    pit_temperature : float
        Temperature [°C] at which a new tyre exits the pit lane.
    """

    name: str

    # Grip / wear model
    base_grip: float
    wear_rate_per_km: float
    min_grip_multiplier: float
    cliff_wear: float
    cliff_penalty: float

    # Temperature model
    optimal_temperature: float
    cold_temperature: float
    overheating_temperature: float
    thermal_sensitivity: float
    heating_rate: float
    cooling_rate: float

    # Wear thermal sensitivity
    cold_wear_multiplier: float
    hot_wear_multiplier: float
    pit_temperature: float

    # Empirical lap-time degradation overlay [s/lap on tyre].
    # Applied on top of physics to reproduce observed performance drop.
    # Calibrated as GROSS rate (compensates for the ~0.085 s/lap fuel-burn
    # speed-up already present in the physics model).
    # Real net observed: Soft ≈ +0.14 s/lap, Medium ≈ +0.085, Hard ≈ +0.045.
    deg_s_per_lap: float = 0.0

    # Compound family — drives wet-weather grip behaviour (see wet_grip_factor).
    #   "slick"        : S / M / H — maximum grip when dry, useless in the wet
    #   "intermediate" : crossover compound for a damp track
    #   "wet"          : full wet for standing water
    compound_type: str = "slick"

    # ------------------------------------------------------------------ #
    # Wet-weather grip                                                     #
    # ------------------------------------------------------------------ #

    def wet_grip_factor(self, wetness: float) -> float:
        """
        Multiplier applied to grip as a function of track wetness [0, 1].

        Models the crossover between compound families:
        - slicks lose grip steeply once the track is wet (no water clearing),
        - intermediates peak on a damp track (~40 % wet) and overheat when dry,
        - full wets need standing water and peak near a soaked track (~85 %).

        Combined with each compound's ``base_grip`` this reproduces the real
        crossover: in light damp slicks still win, past ~30 % wet the
        intermediate is fastest, and on a soaked track the full wet wins.
        """
        w = max(0.0, min(1.0, wetness))
        if self.compound_type == "intermediate":
            return max(0.55, 1.0 - 1.6 * (w - 0.40) ** 2)
        if self.compound_type == "wet":
            return max(0.50, 1.0 - 1.1 * (w - 0.85) ** 2)
        # slick
        return max(0.30, 1.0 - 0.62 * w)

    # ------------------------------------------------------------------ #
    # Wear                                                                 #
    # ------------------------------------------------------------------ #

    def wear_increment(
        self,
        distance_m: float,
        temperature: float | None = None,
    ) -> float:
        """
        Wear increment over a distance step.

        Cold tyres wear slightly more due to sliding; overheated tyres
        wear significantly more due to thermal degradation.
        """
        base_wear = self.wear_rate_per_km * m_to_km(distance_m)

        if temperature is None:
            return base_wear

        multiplier = 1.0

        if temperature < self.cold_temperature:
            delta = self.cold_temperature - temperature
            multiplier += self.cold_wear_multiplier * delta / 20.0

        if temperature > self.overheating_temperature:
            delta = temperature - self.overheating_temperature
            multiplier += self.hot_wear_multiplier * delta / 20.0

        return base_wear * multiplier

    # ------------------------------------------------------------------ #
    # Grip                                                                 #
    # ------------------------------------------------------------------ #

    def wear_grip_multiplier(self, tyre_wear: float) -> float:
        """
        Grip multiplier from wear — three-phase model:

        Phase 1 (0 → cliff_wear):  gentle logarithmic drop, tyres
            are green but performance is acceptable.
        Phase 2 (cliff_wear → 1.0):  accelerated linear cliff —
            thermal blistering / graining kicks in.
        Phase 3 (> 1.0, theoretical):  extrapolated cliff slope,
            never reached in practice but avoids hard floor artefacts.

        The hard floor min_grip_multiplier is retained only as an
        absolute safety bound, not as a normal operating point.
        Realistic Monza Hard grip loss: ~0.06 per 10 laps post-cliff.
        """
        tyre_wear = max(0.0, tyre_wear)

        # Phase 1: logarithmic drop (0 → cliff_wear)
        # Use actual wear (capped at cliff_wear) so grip declines
        # gradually from lap 1, not in a flat pre-cliff plateau.
        pre_cliff_loss = 0.045 * math.log(1.0 + 3.5 * min(tyre_wear, self.cliff_wear))
        grip = self.base_grip - pre_cliff_loss

        # Phase 2: linear cliff beyond cliff_wear (no hard floor)
        if tyre_wear > self.cliff_wear:
            over = tyre_wear - self.cliff_wear
            grip -= self.cliff_penalty * over

        # Absolute floor — only hit if compound is catastrophically worn
        return max(self.min_grip_multiplier, grip)

    def temperature_grip_multiplier(self, temperature: float) -> float:
        """
        Grip multiplier from temperature only.

        Peaks at optimal_temperature; drops symmetrically when cold or
        overheated.
        """
        delta = abs(temperature - self.optimal_temperature)
        loss = self.thermal_sensitivity * delta
        return max(0.85, 1.0 - loss)

    def grip_multiplier(
        self,
        tyre_wear: float,
        temperature: float | None = None,
    ) -> float:
        """
        Combined grip multiplier from wear and temperature.

        If temperature is None, only wear is considered (backward
        compatible with single-tyre simulations).
        """
        wear_grip = self.wear_grip_multiplier(tyre_wear)

        if temperature is None:
            return wear_grip

        thermal_grip = self.temperature_grip_multiplier(temperature)
        return max(self.min_grip_multiplier, wear_grip * thermal_grip)


@dataclass
class TyreState:
    """
    Mutable tyre state for a single axle or a combined tyre model.

    Tracks wear, temperature and distance covered on the current set.
    """

    compound: TyreCompound
    wear: float = 0.0
    temperature: float = 70.0
    distance_on_tyre: float = 0.0

    # Thermal limits
    min_temperature: float = 40.0
    max_temperature: float = 140.0
    ambient_temperature: float = 25.0

    # Track wetness [0 = dry, 1 = soaked] — Level A static weather model.
    track_wetness: float = 0.0

    # ------------------------------------------------------------------ #
    # Grip                                                                 #
    # ------------------------------------------------------------------ #

    def grip_multiplier(self) -> float:
        """Combined grip from wear, temperature, warm-up and track wetness."""
        base = self.compound.grip_multiplier(
            tyre_wear=self.wear,
            temperature=self.temperature,
        )
        wet = self.compound.wet_grip_factor(self.track_wetness)
        return base * self._warmup_multiplier() * wet

    def _warmup_multiplier(self) -> float:
        """
        Exponential warm-up curve.

        New tyres ramp up to full grip over the first ~1.5 km.
        """
        distance_km = m_to_km(self.distance_on_tyre)
        warmup = 1.0 - 0.06 * math.exp(-distance_km / 1.5)
        return min(1.0, warmup)

    # ------------------------------------------------------------------ #
    # State update                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        distance_m: float,
        speed: float,
        acceleration: float,
        grip_usage: float,
        curvature: float = 0.0,
    ) -> None:
        """
        Advance tyre state over a spatial step.

        Parameters
        ----------
        distance_m : float
            Step length [m].
        speed : float
            Vehicle speed [m/s].
        acceleration : float
            Longitudinal acceleration [m/s²].
        grip_usage : float
            Fractional grip demand [0, 1].
        curvature : float
            Track curvature [1/m] at this step.
        """
        ds_km = m_to_km(distance_m)
        self.distance_on_tyre += distance_m

        # --- Temperature heating ---
        load_factor      = max(0.0, min(1.0, grip_usage))
        accel_factor     = min(abs(acceleration) / 20.0, 1.5)
        speed_factor     = min(speed / 80.0, 1.5)
        cornering_factor = min(curvature * 120.0, 2.0)

        thermal_excess     = max(0.0, self.temperature - self.compound.optimal_temperature)
        thermal_saturation = max(0.35, 1.0 - thermal_excess / 80.0)

        heating = (
            self.compound.heating_rate
            * ds_km
            * (
                0.25 * load_factor
                + 0.20 * accel_factor
                + 0.10 * speed_factor
                + 0.12 * cornering_factor
            )
            * thermal_saturation
            * 100.0
        )

        # --- Temperature cooling ---
        # Airflow capped at 1.0: at >252 km/h the boundary layer saturates and
        # convective cooling no longer scales linearly with speed.
        cooling_airflow = min(max(speed / 70.0, 0.25), 1.0)

        # Cooling floor ≈ cold_temperature - 5°C: represents the minimum
        # temperature the tyre can reach from rolling-resistance heat generation
        # and road heat soak, even on the longest high-speed straight.
        cooling_floor = max(self.ambient_temperature, self.compound.cold_temperature - 5.0)

        cooling = (
            self.compound.cooling_rate
            * ds_km
            * cooling_airflow
            * max(0.0, self.temperature - cooling_floor)
            * 1.10
        )

        self.temperature = max(
            self.min_temperature,
            min(self.max_temperature, self.temperature + heating - cooling),
        )

        # --- Wear update ---
        # No upper cap: wear > 1.0 models a tyre run beyond its design limit.
        # wear_grip_multiplier continues to decline past 1.0 via the cliff
        # penalty, so lap times keep increasing rather than flattening out.
        # The display layer shows min(wear, 1.0)*100 as a percentage, but the
        # physics uses the raw value.
        self.wear = max(
            0.0,
            self.wear + self.compound.wear_increment(
                distance_m=distance_m,
                temperature=self.temperature,
            ),
        )

    def reset(
        self,
        compound: TyreCompound | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Reset state for a new set of tyres (pit stop).

        Parameters
        ----------
        compound : TyreCompound | None
            New compound to fit. If None, keeps the current compound.
        temperature : float | None
            Starting temperature. Defaults to compound.pit_temperature.
        """
        if compound is not None:
            self.compound = compound

        self.wear = 0.0
        self.distance_on_tyre = 0.0
        self.temperature = (
            temperature
            if temperature is not None
            else self.compound.pit_temperature
        )


# ------------------------------------------------------------------ #
# Compound definitions                                                #
# ------------------------------------------------------------------ #

SOFT = TyreCompound(
    name="Soft",
    # Grip / wear — Soft reaches the cliff around 18 laps at Monza
    base_grip=1.03,
    wear_rate_per_km=0.0072,         # 0.0072/km × 5.793 km ≈ 0.042/lap
    min_grip_multiplier=0.20,
    cliff_wear=0.75,                 # cliff at ~75% wear (~lap 18)
    cliff_penalty=0.30,
    # Temperature — warm-up 1-2 laps, overheat risk on long stints
    optimal_temperature=92.0,
    cold_temperature=72.0,
    overheating_temperature=108.0,
    thermal_sensitivity=0.0028,
    heating_rate=1.80,
    cooling_rate=1.10,
    cold_wear_multiplier=0.22,
    hot_wear_multiplier=0.85,
    pit_temperature=68.0,
    # Gross degradation overlay — calibrated so 1-stop vs 2-stop gap ≈ 15s at Monza
    # Net observed: ~0.10 s/lap (real ~0.14 s/lap; gap closed by SC/traffic in real races)
    deg_s_per_lap=0.145,
)

MEDIUM = TyreCompound(
    name="Medium",
    # Grip / wear — Medium reaches the cliff around 30 laps at Monza
    base_grip=1.00,
    wear_rate_per_km=0.0043,         # 0.0043/km × 5.793 km ≈ 0.025/lap
    min_grip_multiplier=0.20,
    cliff_wear=0.75,                 # cliff at ~75% wear (~lap 30)
    cliff_penalty=0.22,
    # Temperature
    optimal_temperature=98.0,
    cold_temperature=76.0,
    overheating_temperature=116.0,
    thermal_sensitivity=0.0022,
    heating_rate=1.50,
    cooling_rate=0.95,
    cold_wear_multiplier=0.18,
    hot_wear_multiplier=0.65,
    pit_temperature=70.0,
    # Gross degradation overlay — net observed ~0.07 s/lap (real ~0.085 s/lap)
    deg_s_per_lap=0.105,
)

HARD = TyreCompound(
    name="Hard",
    # Grip / wear — Hard reaches the cliff around 40-42 laps at Monza
    base_grip=0.97,
    wear_rate_per_km=0.0031,         # 0.0031/km × 5.793 km ≈ 0.018/lap
    min_grip_multiplier=0.20,
    cliff_wear=0.75,                 # cliff at ~75% wear (~lap 42)
    cliff_penalty=0.18,
    # Temperature — Hard needs 3-4 laps to get into window
    optimal_temperature=104.0,
    cold_temperature=80.0,
    overheating_temperature=122.0,
    thermal_sensitivity=0.0018,
    heating_rate=1.20,
    cooling_rate=0.80,
    cold_wear_multiplier=0.14,
    hot_wear_multiplier=0.50,
    pit_temperature=68.0,
    # Gross degradation overlay — net observed ~0.04 s/lap (real ~0.04-0.05 s/lap)
    deg_s_per_lap=0.078,
)

INTERMEDIATE = TyreCompound(
    name="Intermediate",
    # Crossover compound: lower ultimate grip than a slick, but works on a
    # damp track. base_grip × wet_grip_factor peaks around 40 % wetness.
    # base_grip calibrated vs Silverstone 2024 (HAM): inter pace +10.8 s over
    # the dry slick lap, matching the real +10-12 s on the intermediate stint.
    base_grip=0.75,
    wear_rate_per_km=0.0050,
    min_grip_multiplier=0.20,
    cliff_wear=0.80,
    cliff_penalty=0.20,
    # Temperature — inters run cooler than slicks; overheat on a drying line.
    optimal_temperature=70.0,
    cold_temperature=50.0,
    overheating_temperature=95.0,
    thermal_sensitivity=0.0026,
    heating_rate=1.40,
    cooling_rate=1.30,
    cold_wear_multiplier=0.20,
    hot_wear_multiplier=1.20,        # cooks quickly when the track dries
    pit_temperature=55.0,
    deg_s_per_lap=0.120,
    compound_type="intermediate",
)

WET = TyreCompound(
    name="Wet",
    # Full wet: most water displacement, lowest ultimate grip, needs standing
    # water. base_grip × wet_grip_factor peaks around 85 % wetness.
    # No full-wet real data from Silverstone 2024 (inters only); base_grip set
    # below the intermediate (same ~0.91 ratio) for a plausible +15-18 s pace.
    base_grip=0.68,
    wear_rate_per_km=0.0040,
    min_grip_multiplier=0.20,
    cliff_wear=0.82,
    cliff_penalty=0.18,
    optimal_temperature=60.0,
    cold_temperature=42.0,
    overheating_temperature=85.0,
    thermal_sensitivity=0.0024,
    heating_rate=1.25,
    cooling_rate=1.45,
    cold_wear_multiplier=0.18,
    hot_wear_multiplier=1.40,
    pit_temperature=50.0,
    deg_s_per_lap=0.095,
    compound_type="wet",
)

TYRE_COMPOUNDS: dict[str, TyreCompound] = {
    "soft":         SOFT,
    "medium":       MEDIUM,
    "hard":         HARD,
    "intermediate": INTERMEDIATE,
    "wet":          WET,
}
