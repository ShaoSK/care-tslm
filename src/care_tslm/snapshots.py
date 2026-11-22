"""Cut one event into multi-scale snapshots (PLAN.md 5.1, 5.3, 5.5).

A snapshot ends at step ``t`` (inclusive) and holds, for every channel:

- long:  the 30 days up to t, as 180 means of 4 hours;
- short: the 7.5 days up to t, as 180 means of 1 hour.

Component channels carry residual z; context channels (ambient, power fraction, wind) carry raw values.
The label belongs to the end time t. No snapshot reaches into the baseline, because the residual
model was fitted there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from care_tslm.config import (
    LOAD_BINS,
    PAIR_ROLE,
    Z_CLIP,
    base_sensor,
    channel_list,
    sensor_pairs,
    LONG_BIN,
    LONG_POINTS,
    LONG_STEPS,
    PREDICTION_STRIDE_HOURS,
    SHORT_BIN,
    SHORT_POINTS,
    SHORT_STEPS,
    STEPS_PER_DAY,
    STEPS_PER_HOUR,
    TRAIN_NORMAL_STRIDE_HOURS,
)
from care_tslm.events import EventData
from care_tslm.residuals import Residuals

NORMAL_STATUS = (0, 2)


@dataclass
class Snapshots:
    """All snapshots of one event."""

    channels: list[str]
    """Channel order of the arrays: the sensor names; the power channel holds the power fraction."""
    roles: list[str]
    long: np.ndarray
    """float16 (n, channels, LONG_POINTS); NaN where a bin has no data."""
    short: np.ndarray
    """float16 (n, channels, SHORT_POINTS)."""
    meta: list[dict] = field(default_factory=list)


def _cumulative(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums of values and of finite counts, shape (channels, n_steps + 1)."""
    finite = np.isfinite(series)
    sums = np.concatenate([np.zeros((series.shape[0], 1)), np.cumsum(np.where(finite, series, 0.0), axis=1)], axis=1)
    counts = np.concatenate([np.zeros((series.shape[0], 1)), np.cumsum(finite, axis=1)], axis=1)
    return sums, counts


def _bins(sums: np.ndarray, counts: np.ndarray, ends: np.ndarray, steps: int, width: int) -> np.ndarray:
    """Bin means of the ``steps`` before each exclusive end: (n, channels, steps // width)."""
    edges = ends[:, None] - steps + np.arange(0, steps // width + 1) * width  # (n, points + 1)
    total = sums[:, edges[:, 1:]] - sums[:, edges[:, :-1]]  # (channels, n, points)
    count = counts[:, edges[:, 1:]] - counts[:, edges[:, :-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return means.transpose(1, 0, 2)


def _cumulative_sq(series: np.ndarray) -> np.ndarray:
    """Prefix sums of squares, for a window standard deviation (shape (channels, n_steps + 1))."""
    finite = np.isfinite(series)
    squared = np.where(finite, series, 0.0) ** 2
    return np.concatenate([np.zeros((series.shape[0], 1)), np.cumsum(squared, axis=1)], axis=1)


def _cumulative_ty(series: np.ndarray) -> np.ndarray:
    """Prefix sums of step-index * value, for a least-squares slope over a window."""
    finite = np.isfinite(series)
    t = np.arange(series.shape[1], dtype=np.float64)
    return np.concatenate(
        [np.zeros((series.shape[0], 1)), np.cumsum(np.where(finite, series, 0.0) * t, axis=1)], axis=1
    )


def _cumulative_t(finite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums of the step index and its square over the FINITE steps of each channel."""
    t = np.arange(finite.shape[1], dtype=np.float64)
    ts = np.where(finite, t, 0.0)
    tsq = np.where(finite, t * t, 0.0)
    zero = np.zeros((finite.shape[0], 1))
    return (
        np.concatenate([zero, np.cumsum(ts, axis=1)], axis=1),
        np.concatenate([zero, np.cumsum(tsq, axis=1)], axis=1),
    )


def _window_std(
    sums: np.ndarray, sq_sums: np.ndarray, counts: np.ndarray, ends: np.ndarray, steps: int
) -> np.ndarray:
    """Standard deviation over the ``steps`` before each exclusive end: (channels, n)."""
    total = sums[:, ends] - sums[:, ends - steps]
    squared = sq_sums[:, ends] - sq_sums[:, ends - steps]
    count = counts[:, ends] - counts[:, ends - steps]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        variance = np.where(count > 1, squared / np.maximum(count, 1) - mean**2, np.nan)
    return np.sqrt(np.maximum(variance, 0.0))


def _window_slope(
    sums: np.ndarray,
    ty_sums: np.ndarray,
    t_sums: np.ndarray,
    tsq_sums: np.ndarray,
    counts: np.ndarray,
    ends: np.ndarray,
    steps: int,
    per_day: float,
) -> np.ndarray:
    """Least-squares slope per day over the ``steps`` before each exclusive end: (channels, n).

    ``cov(t, y) / var(t)`` from prefix sums, so a 30-day trend costs no more than a mean.
    """
    window = lambda c: c[:, ends] - c[:, ends - steps]  # noqa: E731
    n = window(counts)
    sy, sty, st, stt = window(sums), window(ty_sums), window(t_sums), window(tsq_sums)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sty / np.maximum(n, 1) - (st / np.maximum(n, 1)) * (sy / np.maximum(n, 1))
        var = stt / np.maximum(n, 1) - (st / np.maximum(n, 1)) ** 2
        slope = np.where((n > 20) & (var > 0), cov / np.where(var > 0, var, 1), np.nan)
    return slope * per_day


def _window_mean(sums: np.ndarray, counts: np.ndarray, ends: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and finite fraction of the ``steps`` before each exclusive end: (channels, n) each."""
    total = sums[:, ends] - sums[:, ends - steps]
    count = counts[:, ends] - counts[:, ends - steps]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, count / steps


def end_steps(event: EventData) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot end steps: (training-normal ends, prediction ends)."""
    first_allowed = event.baseline_end + LONG_STEPS - 1
    train_last = event.train_steps[1]
    stride = TRAIN_NORMAL_STRIDE_HOURS * STEPS_PER_HOUR
    # Align training ends to the end of the training period, walking backwards.
    train_ends = np.arange(train_last, first_allowed - 1, -stride)[::-1]
    p_first, p_last = event.prediction_steps
    p_stride = PREDICTION_STRIDE_HOURS * STEPS_PER_HOUR
    pred_ends = np.arange(max(p_first, first_allowed), p_last + 1, p_stride)
    return train_ends.astype(np.int64), pred_ends.astype(np.int64)


def pair_channel(residuals: Residuals, a: str, b: str) -> np.ndarray:
    """``(z_a - z_b - mean) / std`` with the healthy baseline's difference statistics, clipped.

    All NaN when the baseline had too few points for the pair (``residuals.pair_baseline`` has no
    entry): an unknown spread cannot be put on the common scale."""
    stats = residuals.pair_baseline.get(f"{a}-{b}")
    if not stats or not stats.get("std"):
        return np.full_like(residuals.z[a], np.nan)
    diff = (residuals.z[a].astype(np.float64) - residuals.z[b] - stats["mean"]) / stats["std"]
    return np.clip(diff, -Z_CLIP, Z_CLIP).astype(np.float32)


def build(event: EventData, residuals: Residuals, *, with_raw: bool = False, with_pairs: bool = False) -> Snapshots:
    """Every snapshot of one event, with arrays, labels and summary statistics.

    ``with_raw`` carries every component channel a second time as the raw sensor reading
    (PLAN.md 5.4): the residual says how far from normal, the raw curve keeps the absolute level
    and shape that the residual's conditional fit takes out.
    """
    channels = channel_list(event.farm, with_raw=with_raw, with_pairs=with_pairs)
    names = [c.sensor for c in channels]
    rows = []
    for c in channels:
        if c.role == PAIR_ROLE:
            a, b = c.sensor.split("-")
            rows.append(pair_channel(residuals, a, b))
        elif c.role == "component":
            rows.append(residuals.z[c.sensor])
        elif c.role == "component_raw":
            rows.append(event.channels[base_sensor(c.sensor)])
        elif c.role == "power":
            rows.append(residuals.power_fraction)
        else:
            rows.append(event.channels[c.sensor])
    series = np.stack(rows).astype(np.float64)
    sums, counts = _cumulative(series)
    # The component channels' own readings, for the "physical value vs healthy baseline" rationale.
    raw_series = np.stack(
        [event.channels[base_sensor(c.sensor)] for c in channels if c.role == "component"]
    ).astype(np.float64)
    raw_series = np.where(residuals.operating[None, :], raw_series, np.nan)
    raw_sums, raw_counts = _cumulative(raw_series)
    raw_sq = _cumulative_sq(raw_series)
    # Trend of the residual (not of the raw reading): a slope in sigma per day is comparable
    # across sensors, a slope in degrees per day is not.
    z_series = np.stack([residuals.z[base_sensor(c.sensor)] for c in channels if c.role == "component"]).astype(
        np.float64
    )
    z_sums, z_counts = _cumulative(z_series)
    z_ty = _cumulative_ty(z_series)
    z_t, z_tsq = _cumulative_t(np.isfinite(z_series))
    op_sums, op_counts = _cumulative(residuals.operating[None, :].astype(np.float64))

    train_ends, pred_ends = end_steps(event)
    ends_incl = np.concatenate([train_ends, pred_ends])
    ends = ends_incl + 1  # exclusive
    n = len(ends)
    if n == 0:
        empty = np.zeros((0, len(names), 0), dtype=np.float16)
        return Snapshots(channels=names, roles=[c.role for c in channels], long=empty, short=empty)

    long = _bins(sums, counts, ends, LONG_STEPS, LONG_BIN).astype(np.float16)
    short = _bins(sums, counts, ends, SHORT_STEPS, SHORT_BIN).astype(np.float16)
    assert long.shape[2] == LONG_POINTS and short.shape[2] == SHORT_POINTS

    week = 7 * STEPS_PER_DAY
    stats = {
        "mean_30d": _window_mean(sums, counts, ends, LONG_STEPS)[0],
        "mean_7d": _window_mean(sums, counts, ends, week)[0],
        "mean_24h": _window_mean(sums, counts, ends, STEPS_PER_DAY)[0],
    }
    coverage_7d = _window_mean(sums, counts, ends, week)[1]
    raw_stats = {
        "raw_mean_30d": _window_mean(raw_sums, raw_counts, ends, LONG_STEPS)[0],
        "raw_mean_7d": _window_mean(raw_sums, raw_counts, ends, week)[0],
        "raw_mean_24h": _window_mean(raw_sums, raw_counts, ends, STEPS_PER_DAY)[0],
        # Variability and trend (PLAN.md 6): the teacher study found the discriminating signal is
        # often not in the mean -- a bearing that spikes intermittently, or a cell temperature whose
        # 7-day slope is ten times its 30-day slope.
        "raw_std_30d": _window_std(raw_sums, raw_sq, raw_counts, ends, LONG_STEPS),
        "raw_std_7d": _window_std(raw_sums, raw_sq, raw_counts, ends, week),
        "slope_30d": _window_slope(z_sums, z_ty, z_t, z_tsq, z_counts, ends, LONG_STEPS, STEPS_PER_DAY),
        "slope_7d": _window_slope(z_sums, z_ty, z_t, z_tsq, z_counts, ends, week, STEPS_PER_DAY),
    }
    operating_7d = _window_mean(op_sums, op_counts, ends, week)[0][0]
    operating_24h = _window_mean(op_sums, op_counts, ends, STEPS_PER_DAY)[0][0]

    # Pair differences and load-banded means (PLAN.md 8.3 findings 1 and 4). Both reuse the same
    # prefix sums, so a 30-day window costs no more than a mean.
    pair_names = [f"{a}-{b}" for _, a, b in sensor_pairs(event.farm)]
    pair_stats: dict[str, np.ndarray] = {}
    if pair_names:
        pair_series = np.stack(
            [residuals.z[a] - residuals.z[b] for _, a, b in sensor_pairs(event.farm)]
        ).astype(np.float64)
        p_sums, p_counts = _cumulative(pair_series)
        p_sq = _cumulative_sq(pair_series)
        pair_stats = {
            "mean_7d": _window_mean(p_sums, p_counts, ends, week)[0],
            "std_7d": _window_std(p_sums, p_sq, p_counts, ends, week),
            "mean_30d": _window_mean(p_sums, p_counts, ends, LONG_STEPS)[0],
        }
    band_stats: dict[str, np.ndarray] = {}
    for band, lo, hi in LOAD_BINS:
        inside = (residuals.power_fraction > lo) & (residuals.power_fraction <= hi)
        banded = np.where(inside[None, :], z_series, np.nan)
        b_sums, b_counts = _cumulative(banded)
        band_stats[band] = _window_mean(b_sums, b_counts, ends, week)[0]

    w_first, w_last = event.window_steps
    component_index = [i for i, c in enumerate(channels) if c.role == "component"]
    meta = []
    for k in range(n):
        t = int(ends_incl[k])
        role = "train_normal" if k < len(train_ends) else "prediction"
        in_window = role == "prediction" and w_first <= t <= w_last
        target = "anomaly" if (event.label == "anomaly" and in_window) else "normal"
        if role == "train_normal" or event.label == "normal":
            pool = "normal"
        elif in_window:
            pool = "anomaly"
        else:
            pool = None  # anomaly event, outside the window: scored, never trained on (label is uncertain)
        status_end = int(event.status[t])
        meta.append(
            {
                "farm": event.farm,
                "event_id": event.event_id,
                "turbine": event.turbine,
                "event_label": event.label,
                "index": k,
                "end_step": t,
                "end_us": event.timestamp_us(t),
                "role": role,
                "in_window": bool(in_window),
                "target": target,
                "train_pool": pool,
                "status_end": status_end,
                "scored_normal_status": status_end in NORMAL_STATUS,
                "operating_7d": round(float(operating_7d[k]), 3),
                "operating_24h": round(float(operating_24h[k]), 3),
                "pairs": {
                    name: {
                        key: (None if not np.isfinite(values[j, k]) else round(float(values[j, k]), 2))
                        for key, values in pair_stats.items()
                    }
                    for j, name in enumerate(pair_names)
                },
                "components": {
                    names[i]: {
                        name: (None if not np.isfinite(values[i, k]) else round(float(values[i, k]), 3))
                        for name, values in stats.items()
                    }
                    | {
                        name: (None if not np.isfinite(values[j, k]) else round(float(values[j, k]), 2))
                        for name, values in raw_stats.items()
                    }
                    | {
                        f"load_{band}": (
                            None if not np.isfinite(values[j, k]) else round(float(values[j, k]), 2)
                        )
                        for band, values in band_stats.items()
                    }
                    | {"coverage_7d": round(float(coverage_7d[i, k]), 3)}
                    for j, i in enumerate(component_index)
                },
            }
        )
    return Snapshots(channels=names, roles=[c.role for c in channels], long=long, short=short, meta=meta)
