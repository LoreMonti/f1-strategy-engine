"""Unit tests for the Safety-Car history estimator (pure functions, no network)."""
from src.data.sc_history import _count_blocks, _shrink_duration, _PRIOR_SC_DURATION


def test_count_blocks_counts_contiguous_runs():
    flags = [False, True, True, False, True, False, False, True, True, True]
    n, lengths = _count_blocks(flags)
    assert n == 3
    assert lengths == [2, 1, 3]


def test_count_blocks_handles_trailing_run():
    n, lengths = _count_blocks([False, True, True])
    assert n == 1 and lengths == [2]


def test_count_blocks_empty():
    n, lengths = _count_blocks([False, False, False])
    assert n == 0 and lengths == []


def test_shrink_duration_pulls_small_sample_towards_prior():
    # One long 12-lap SC should be pulled well below 12 towards the ~4-lap prior.
    shrunk = _shrink_duration([12], _PRIOR_SC_DURATION)
    assert _PRIOR_SC_DURATION < shrunk < 12


def test_shrink_duration_empty_returns_prior():
    assert _shrink_duration([], _PRIOR_SC_DURATION) == _PRIOR_SC_DURATION


def test_shrink_duration_large_sample_trusts_data():
    # Many consistent observations → estimate close to the data mean.
    shrunk = _shrink_duration([5] * 30, _PRIOR_SC_DURATION)
    assert abs(shrunk - 5.0) < 0.2
