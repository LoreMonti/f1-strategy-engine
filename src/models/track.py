# =========================================================
# Track model
#
# Defines the circuit as a sequence of segments, each with
# geometric and grip properties.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrackSegment:
    """
    A single section of the circuit.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. "Main Straight", "Turn 1").
    length : float
        Segment length [m]. Must be positive.
    curvature : float
        Inverse of the corner radius [1/m].
        0.0 for straights; positive for corners.
    grip_factor : float
        Surface grip multiplier relative to nominal (1.0 = nominal).
    banking_deg : float
        Banking angle [°]. Positive = banked inward.
    traction_factor : float
        Traction multiplier at corner exit (1.0 = nominal).
    braking_severity : float
        Braking demand multiplier at corner entry (1.0 = nominal).
    """

    name: str
    length: float
    curvature: float

    grip_factor: float = 1.0
    banking_deg: float = 0.0
    traction_factor: float = 1.0
    braking_severity: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(
                f"Segment '{self.name}': length must be positive, got {self.length}."
            )
        if self.curvature < 0.0:
            raise ValueError(
                f"Segment '{self.name}': curvature must be >= 0, got {self.curvature}."
            )
        if self.grip_factor <= 0.0:
            raise ValueError(
                f"Segment '{self.name}': grip_factor must be positive, got {self.grip_factor}."
            )

    @property
    def is_straight(self) -> bool:
        return self.curvature == 0.0

    @property
    def corner_radius(self) -> float | None:
        """Corner radius [m], or None for straights."""
        if self.is_straight:
            return None
        return 1.0 / self.curvature


@dataclass
class Track:
    """
    A circuit defined as an ordered sequence of segments.

    Parameters
    ----------
    name : str
        Circuit name (e.g. "Monza", "Spa-Francorchamps").
    segments : list[TrackSegment]
        Ordered list of track segments.
    country : str, optional
        Country where the circuit is located.
    """

    name: str
    segments: list[TrackSegment] = field(default_factory=list)
    country: str = ""

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(f"Track '{self.name}' must have at least one segment.")

    @property
    def total_length(self) -> float:
        """Total circuit length [m]."""
        return sum(segment.length for segment in self.segments)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def corner_segments(self) -> list[TrackSegment]:
        return [s for s in self.segments if not s.is_straight]

    @property
    def straight_segments(self) -> list[TrackSegment]:
        return [s for s in self.segments if s.is_straight]

    def segment_by_name(self, name: str) -> TrackSegment:
        """Return segment by name, raising KeyError if not found."""
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(f"No segment named '{name}' in track '{self.name}'.")

    def __repr__(self) -> str:
        return (
            f"Track(name='{self.name}', "
            f"length={self.total_length:.0f} m, "
            f"segments={self.num_segments})"
        )
