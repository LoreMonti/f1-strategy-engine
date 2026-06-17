# =========================================================
# Race strategy model
#
# Defines pit stops, stints, race strategies and — importantly —
# typed result dataclasses so simulation outputs are not plain dicts.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field

from src.models.tyre import TyreCompound


# ------------------------------------------------------------------ #
# Strategy building blocks                                            #
# ------------------------------------------------------------------ #

@dataclass
class PitStop:
    """
    A single pit stop event.

    Parameters
    ----------
    lap : int
        Lap on which the pit stop occurs (at the start of the lap).
    new_compound : TyreCompound
        Compound fitted after the stop.
    time_loss : float
        Fixed time penalty for the stop [s].
    """
    lap: int
    new_compound: TyreCompound
    time_loss: float = 22.0


@dataclass
class Stint:
    """
    A continuous run on the same tyre compound.

    Parameters
    ----------
    compound : TyreCompound
        Compound used during this stint.
    start_lap : int
        First lap of the stint (1-indexed).
    end_lap : int
        Last lap of the stint (inclusive).
    """
    compound: TyreCompound
    start_lap: int
    end_lap: int

    @property
    def length(self) -> int:
        """Number of laps in the stint."""
        return self.end_lap - self.start_lap + 1

    def contains_lap(self, lap: int) -> bool:
        return self.start_lap <= lap <= self.end_lap


@dataclass
class RaceStrategy:
    """
    A complete race strategy: starting compound + ordered pit stops.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Soft-Medium L4").
    initial_compound : TyreCompound
        Compound at race start.
    pit_stops : list[PitStop]
        Ordered list of pit stop events.
    """
    name: str
    initial_compound: TyreCompound
    pit_stops: list[PitStop] = field(default_factory=list)

    def pit_stop_on_lap(self, lap: int) -> PitStop | None:
        """Return the pit stop scheduled on this lap, or None."""
        for pit in self.pit_stops:
            if pit.lap == lap:
                return pit
        return None

    def build_stints(self, num_laps: int) -> list[Stint]:
        """Derive the stint sequence from pit stop events."""
        stints: list[Stint] = []
        compound = self.initial_compound
        start = 1

        for pit in sorted(self.pit_stops, key=lambda p: p.lap):
            if pit.lap - 1 >= start:
                stints.append(Stint(compound, start, pit.lap - 1))
            compound = pit.new_compound
            start = pit.lap

        if start <= num_laps:
            stints.append(Stint(compound, start, num_laps))

        return stints


# ------------------------------------------------------------------ #
# Typed result dataclasses                                            #
# ------------------------------------------------------------------ #

@dataclass
class LapResult:
    """
    Per-lap result from a race simulation.

    This replaces the plain dict used in the original code, giving
    attribute access, type hints, and IDE autocompletion.
    """
    lap: int
    compound: str

    # Timing
    raw_lap_time: float         # pure lap time, no pit [s]
    pit_time_loss: float        # time lost in pit stop this lap [s]
    lap_time: float             # raw + pit loss [s]
    delta_lap_time: float       # vs previous lap [s]
    cumulative_time: float      # total elapsed race time [s]

    # Pit flag
    pit_stop: bool

    # Tyre state at lap end
    final_tyre_wear: float
    final_tyre_temperature: float
    final_grip_multiplier: float
    final_front_tyre_wear: float
    final_rear_tyre_wear: float
    final_front_tyre_temperature: float
    final_rear_tyre_temperature: float

    # Fuel / mass
    initial_fuel_mass: float
    final_fuel_mass: float
    fuel_used: float
    final_vehicle_mass: float

    # Speed
    max_speed: float            # [m/s]
    final_speed: float          # [m/s]
    final_gear: int


@dataclass
class RaceResult:
    """
    Full race simulation result for one strategy.
    """
    track: str
    vehicle: str
    strategy: str
    num_laps: int

    # Aggregates
    total_time: float
    total_pit_loss: float
    average_raw_lap_time: float

    # Per-lap breakdown
    laps: list[LapResult]

    # Derived
    fastest_lap: LapResult      # lap with minimum raw_lap_time
    stints: list[Stint]

    @property
    def num_stops(self) -> int:
        return sum(1 for lap in self.laps if lap.pit_stop)
