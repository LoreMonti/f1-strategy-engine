# =========================================================
# Multi-car race result model
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CarLapResult:
    """Per-lap result for one car in a multi-car race."""
    lap: int
    compound: str

    # Timing
    raw_lap_time: float       # base lap time (tyre/fuel model, no traffic)
    traffic_penalty: float    # seconds lost behind slower car
    pit_time_loss: float      # pit stop delta
    lap_time: float           # raw + traffic + pit
    cumulative_time: float    # total elapsed since lights out (incl. grid gap)

    # Race position
    position: int
    gap_to_leader: float      # seconds behind P1 at end of this lap

    # Flags
    pit_stop: bool

    # Tyre state
    final_tyre_wear: float
    final_tyre_temperature: float
    final_grip_multiplier: float


@dataclass
class CarRaceResult:
    """Full race result for one car."""
    name: str
    strategy_name: str
    grid_position: int           # 1-indexed starting position
    grid_gap_s: float            # starting time handicap [s]

    final_position: int
    total_time: float            # cumulative time at chequered flag
    total_traffic_penalty: float

    laps: list[CarLapResult] = field(default_factory=list)

    @property
    def num_laps(self) -> int:
        return len(self.laps)

    @property
    def fastest_lap(self) -> CarLapResult:
        return min(self.laps, key=lambda lr: lr.raw_lap_time)


@dataclass
class MultiCarRaceResult:
    """Aggregated result for a full multi-car race."""
    cars: list[CarRaceResult]    # ordered by final_position
    num_laps: int
    track: str

    @property
    def winner(self) -> CarRaceResult:
        return self.cars[0]
