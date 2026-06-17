"""Tests for the WeatherModel (Level A static + Level B dynamic)."""
import math

from src.models.weather import WeatherModel


def test_constant_is_flat_and_not_dynamic():
    w = WeatherModel.constant(0.6)
    assert w.wetness(1) == 0.6
    assert w.wetness(50) == 0.6
    assert not w.is_dynamic
    assert w.max_wetness == 0.6


def test_constant_clamped_to_unit_interval():
    assert WeatherModel.constant(1.5).wetness(1) == 1.0
    assert WeatherModel.constant(-0.2).wetness(1) == 0.0


def test_linear_interpolation_between_keyframes():
    w = WeatherModel.from_keyframes([(1, 0.0), (11, 1.0)])
    assert math.isclose(w.wetness(6), 0.5, abs_tol=1e-9)   # midpoint
    assert math.isclose(w.wetness(1), 0.0)
    assert math.isclose(w.wetness(11), 1.0)


def test_clamped_outside_keyframe_range():
    w = WeatherModel.from_keyframes([(10, 0.4), (20, 0.8)])
    assert w.wetness(1) == 0.4     # before first → held flat
    assert w.wetness(50) == 0.8    # after last → held flat


def test_dynamic_detection_and_peak():
    w = WeatherModel.from_keyframes(
        [{"lap": 1, "wetness": 0.0}, {"lap": 10, "wetness": 0.6},
         {"lap": 20, "wetness": 0.0}]
    )
    assert w.is_dynamic
    assert math.isclose(w.max_wetness, 0.6)
