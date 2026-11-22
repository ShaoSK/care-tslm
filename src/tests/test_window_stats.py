"""The prefix-sum window statistics: standard deviation and least-squares slope (PLAN.md 6)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.snapshots import (  # noqa: E402
    _cumulative,
    _cumulative_sq,
    _cumulative_t,
    _cumulative_ty,
    _window_slope,
    _window_std,
)


def std_of(series: np.ndarray, steps: int, end: int) -> float:
    sums, counts = _cumulative(series)
    sq = _cumulative_sq(series)
    return float(_window_std(sums, sq, counts, np.array([end + 1]), steps)[0, 0])


def slope_of(series: np.ndarray, steps: int, end: int, per_day: float) -> float:
    sums, counts = _cumulative(series)
    ty = _cumulative_ty(series)
    t, tsq = _cumulative_t(np.isfinite(series))
    return float(_window_slope(sums, ty, t, tsq, counts, np.array([end + 1]), steps, per_day)[0, 0])


def test_the_window_std_matches_numpy():
    rng = np.random.default_rng(0)
    series = rng.normal(5.0, 2.0, size=(1, 400))
    assert std_of(series, 100, 399) == np.float32(np.std(series[0, 300:400])).item() or np.isclose(
        std_of(series, 100, 399), np.std(series[0, 300:400]), atol=1e-6
    )


def test_the_window_std_ignores_missing_steps():
    series = np.array([[1.0, np.nan, 1.0, np.nan, 1.0, 3.0]])
    assert np.isclose(std_of(series, 6, 5), np.std([1.0, 1.0, 1.0, 3.0]), atol=1e-9)


def test_a_flat_series_has_zero_std_and_zero_slope():
    series = np.full((1, 300), 7.0)
    assert np.isclose(std_of(series, 288, 299), 0.0, atol=1e-9)
    assert np.isclose(slope_of(series, 288, 299, 144.0), 0.0, atol=1e-9)


def test_the_slope_recovers_a_known_ramp():
    # +0.01 per step over a 144-step day = +1.44 sigma per day.
    series = (0.01 * np.arange(300.0))[None, :]
    assert np.isclose(slope_of(series, 288, 299, 144.0), 1.44, atol=1e-6)


def test_the_slope_is_negative_for_a_falling_series():
    series = (-0.005 * np.arange(300.0))[None, :]
    assert slope_of(series, 288, 299, 144.0) < 0


def test_the_slope_matches_polyfit_on_a_noisy_ramp():
    rng = np.random.default_rng(3)
    t = np.arange(600.0)
    series = (0.004 * t + rng.normal(0, 0.3, size=600))[None, :]
    got = slope_of(series, 288, 599, 144.0)
    want = np.polyfit(t[312:600], series[0, 312:600], 1)[0] * 144.0
    assert np.isclose(got, want, rtol=1e-6)


def test_too_few_points_gives_nan_rather_than_a_wild_slope():
    series = np.full((1, 300), np.nan)
    series[0, -5:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isnan(slope_of(series, 288, 299, 144.0))
