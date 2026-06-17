"""
tests/test_loaders.py
---------------------
Tests for src/data/loaders.py — uses the real Track / TyreCompound models.

Run with:  pytest tests/test_loaders.py -v
"""

import math
import pytest
from pathlib import Path

from src.data.loaders import TrackLoader, TrackLoadError, load_track
from src.models.track import Track, TrackSegment
from src.models.tyre import TyreCompound

MONZA_YAML = Path("data/tracks/monza_2024.yaml")

pytestmark = pytest.mark.skipif(
    not MONZA_YAML.exists(),
    reason=f"Track YAML not found at {MONZA_YAML}",
)


@pytest.fixture(scope="module")
def loader() -> TrackLoader:
    return load_track(MONZA_YAML)


# ── race_info ─────────────────────────────────────────────────────────────────

class TestRaceInfo:
    def test_circuit_id(self, loader):
        assert loader.race_info().circuit_id == "monza"

    def test_race_laps(self, loader):
        assert loader.race_info().race_laps == 53

    def test_lap_distance(self, loader):
        assert math.isclose(loader.race_info().lap_distance_km, 5.793, rel_tol=1e-3)

    def test_pit_delta_positive(self, loader):
        assert loader.race_info().pit_lane_delta_s > 0

    def test_fuel_covers_race(self, loader):
        ri = loader.race_info()
        assert ri.fuel_load_kg >= ri.fuel_consumption_kg_per_lap * ri.race_laps - 5

    def test_sc_probability_range(self, loader):
        ri = loader.race_info()
        assert 0.0 <= ri.sc_probability_per_lap <= 1.0
        assert 0.0 <= ri.vsc_probability_per_lap <= 1.0


# ── environment ───────────────────────────────────────────────────────────────

class TestEnvironment:
    def test_grip_factor_range(self, loader):
        g = loader.environment().grip_factor
        assert 0.5 <= g <= 1.5

    def test_track_temp_positive(self, loader):
        assert loader.environment().track_temp_c > 0

    def test_tyre_stress_range(self, loader):
        ts = loader.environment().tyre_stress_factor
        assert 0.3 <= ts <= 1.5

    def test_brake_stress_range(self, loader):
        bs = loader.environment().brake_stress_factor
        assert 0.5 <= bs <= 3.0


# ── track() — mini_sectors (default) ─────────────────────────────────────────

class TestTrackMiniSectors:
    @pytest.fixture(scope="class")
    def track(self, loader):
        return loader.track(segment_source="mini_sectors")

    def test_returns_track_instance(self, track):
        assert isinstance(track, Track)

    def test_name(self, track):
        assert "Monza" in track.name

    def test_country(self, track):
        assert track.country == "ITA"

    def test_segment_count(self, track):
        # Each of the 20 mini-sectors expands into 1 (straight) or up to 3
        # (corner: approach / apex / exit) track segments.
        assert track.num_segments >= 20

    def test_all_segments_are_track_segments(self, track):
        for seg in track.segments:
            assert isinstance(seg, TrackSegment)

    def test_total_length_approx(self, track):
        # The lap geometry must reproduce the declared length EXACTLY after the
        # segment-length fix (short approach folded into exit, no lost metres).
        assert math.isclose(track.total_length, 5793, rel_tol=1e-4)

    def test_all_lengths_positive(self, track):
        for seg in track.segments:
            assert seg.length > 0, f"{seg.name}: length={seg.length}"

    def test_curvature_non_negative(self, track):
        for seg in track.segments:
            assert seg.curvature >= 0.0, f"{seg.name}: curvature={seg.curvature}"

    def test_grip_factor_positive(self, track):
        for seg in track.segments:
            assert seg.grip_factor > 0.0

    def test_braking_severity_range(self, track):
        # Should be in [0.5, 2.0] per our mapping
        for seg in track.segments:
            assert 0.4 <= seg.braking_severity <= 2.1, \
                f"{seg.name}: braking_severity={seg.braking_severity}"

    def test_traction_factor_range(self, track):
        # Should be in [0.7, 1.3] per our mapping
        for seg in track.segments:
            assert 0.6 <= seg.traction_factor <= 1.4, \
                f"{seg.name}: traction_factor={seg.traction_factor}"

    def test_heavy_braking_zones_exist(self, track):
        # At least some segments should have braking_severity > 1.5
        heavy = [s for s in track.segments if s.braking_severity > 1.5]
        assert len(heavy) >= 2, "Expected heavy braking zones at T1, Ascari, Parabolica"

    def test_full_throttle_zones_exist(self, track):
        # At least some segments should have traction_factor > 1.2 (full throttle)
        full_thr = [s for s in track.segments if s.traction_factor > 1.2]
        assert len(full_thr) >= 4, "Expected several full-throttle segments at Monza"

    def test_mix_of_corners_and_straights(self, track):
        # is_straight requires curvature == 0.0 exactly; mini-sectors always
        # have a small non-zero κ from the lateral_g formula, so we check via
        # threshold instead.  Monza has clear high-κ zones (T1, Ascari, Parabolica)
        # and clear low-κ zones (main straight, back straight).
        HIGH_CURVATURE = 0.010   # clearly a corner
        LOW_CURVATURE  = 0.002   # effectively a straight

        corners   = [s for s in track.segments if s.curvature >= HIGH_CURVATURE]
        straights = [s for s in track.segments if s.curvature <= LOW_CURVATURE]
        assert len(corners)   >= 3, f"Expected ≥3 corner segments, got {len(corners)}"
        assert len(straights) >= 3, f"Expected ≥3 straight segments, got {len(straights)}"


# ── track() — sectors (coarse) ───────────────────────────────────────────────

class TestTrackSectors:
    @pytest.fixture(scope="class")
    def track(self, loader):
        return loader.track(segment_source="sectors")

    def test_returns_track_instance(self, track):
        assert isinstance(track, Track)

    def test_segment_count(self, track):
        assert track.num_segments == 3

    def test_total_length_approx(self, track):
        assert math.isclose(track.total_length, 5793, rel_tol=0.02)

    def test_segment_names_contain_sector_id(self, track):
        names = [seg.name for seg in track.segments]
        assert any("S1" in n for n in names)
        assert any("S2" in n for n in names)
        assert any("S3" in n for n in names)


# ── tyre_compounds() ──────────────────────────────────────────────────────────

class TestTyreCompounds:
    @pytest.fixture(scope="class")
    def compounds(self, loader):
        return loader.tyre_compounds()

    def test_returns_three_compounds(self, compounds):
        assert len(compounds) == 3

    def test_expected_labels(self, compounds):
        assert set(compounds.keys()) == {"Hard", "Medium", "Soft"}

    def test_all_are_tyre_compound_instances(self, compounds):
        for label, tc in compounds.items():
            assert isinstance(tc, TyreCompound), f"{label} is not a TyreCompound"

    def test_soft_highest_base_grip(self, compounds):
        assert compounds["Soft"].base_grip >= compounds["Medium"].base_grip >= \
               compounds["Hard"].base_grip

    def test_hard_lowest_wear_rate(self, compounds):
        assert compounds["Hard"].wear_rate_per_km <= compounds["Medium"].wear_rate_per_km \
               <= compounds["Soft"].wear_rate_per_km

    def test_optimal_temps_in_plausible_range(self, compounds):
        for label, tc in compounds.items():
            assert 70 <= tc.optimal_temperature <= 130, \
                f"{label}: optimal_temperature={tc.optimal_temperature}"

    def test_soft_higher_overheating_risk(self, compounds):
        # Soft has lower overheating_temperature (more fragile)
        assert compounds["Soft"].overheating_temperature <= \
               compounds["Hard"].overheating_temperature

    def test_grip_multiplier_at_full_wear(self, compounds):
        # Even at wear=1.0, grip floor should be > 0
        for label, tc in compounds.items():
            g = tc.grip_multiplier(tyre_wear=1.0)
            assert g > 0, f"{label} grip at full wear = {g}"

    def test_grip_multiplier_decreases_with_wear(self, compounds):
        for label, tc in compounds.items():
            g0 = tc.grip_multiplier(tyre_wear=0.0)
            g1 = tc.grip_multiplier(tyre_wear=0.8)
            assert g0 > g1, f"{label}: grip should decrease with wear"


# ── Track integration: simulate a lap with loader output ─────────────────────

class TestEndToEnd:
    """Verify loader output is directly consumable by Track methods."""

    def test_segment_by_name(self, loader):
        track = loader.track()
        seg = track.segment_by_name("MS01")
        assert isinstance(seg, TrackSegment)
        assert seg.length > 0

    def test_invalid_segment_name_raises(self, loader):
        track = loader.track()
        with pytest.raises(KeyError):
            track.segment_by_name("NONEXISTENT")

    def test_repr_contains_monza(self, loader):
        track = loader.track()
        assert "Monza" in repr(track)

    def test_total_length_property(self, loader):
        track = loader.track()
        manual = sum(s.length for s in track.segments)
        assert math.isclose(track.total_length, manual, rel_tol=1e-9)

    def test_invalid_segment_source_raises(self, loader):
        with pytest.raises(ValueError, match="segment_source"):
            loader.track(segment_source="invalid")  # type: ignore[arg-type]


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            TrackLoader("data/tracks/does_not_exist.yaml")

    def test_missing_top_level_keys(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("metadata:\n  name: test\n")  # missing required keys
        with pytest.raises(TrackLoadError):
            TrackLoader(str(p))

    def test_sector_distance_mismatch(self, tmp_path):
        """YAML where sector distances don't add up to lap_distance_km."""
        import copy
        import yaml as _yaml

        good_path = MONZA_YAML
        with good_path.open() as fh:
            data = _yaml.safe_load(fh)

        # Corrupt the first sector distance
        data["sectors"][0]["distance_m"] = 9999

        bad_path = tmp_path / "bad_distances.yaml"
        bad_path.write_text(_yaml.dump(data))

        with pytest.raises(TrackLoadError, match="Sector distances"):
            TrackLoader(str(bad_path))
