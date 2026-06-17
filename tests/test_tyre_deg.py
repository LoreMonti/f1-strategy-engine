"""Tests for the learned tyre-degradation fitting (synthetic data, no network)."""
import numpy as np

from src.data.tyre_deg import _stint_slope


def test_stint_slope_recovers_known_linear_rate():
    # A stint that degrades exactly 0.10 s/lap (already fuel-corrected).
    age = np.arange(1, 16, dtype=float)
    time_fc = 90.0 + 0.10 * age
    slope = _stint_slope(age, time_fc)
    assert abs(slope - 0.10) < 0.01


def test_stint_slope_robust_to_outliers():
    # Same 0.10 s/lap trend with two traffic-spoiled laps (+3 s) — Theil-Sen
    # should ignore them.
    age = np.arange(1, 16, dtype=float)
    time_fc = 90.0 + 0.10 * age
    time_fc[5] += 3.0
    time_fc[10] += 3.0
    slope = _stint_slope(age, time_fc)
    assert abs(slope - 0.10) < 0.03


def test_stint_slope_too_few_laps_returns_none():
    assert _stint_slope(np.array([1.0, 2.0, 3.0]), np.array([90.0, 90.1, 90.2])) is None
