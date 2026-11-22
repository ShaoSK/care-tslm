"""Turning hourly judgements into the 10-minute alarm series that CARE-Score scores.

Kept out of ``src/eval/care_score.py`` so it can be tested without installing
``energy-fault-detector`` (which drags in TensorFlow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from care_tslm.config import PERIOD_MIN

STAMPS_PER_HOUR = 60 // PERIOD_MIN
IN_FORCE = pd.Timedelta(hours=1)
"""How long one hourly judgement governs, from the moment it was made."""


def naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop the timezone (after converting to UTC).

    Our ``end_us`` is epoch-UTC microseconds and the CARE CSV timestamps are read as UTC by the
    connector, so comparing the two tz-naive is exact.
    """
    return index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index


def alarm_series(rows: list[dict], index: pd.DatetimeIndex) -> tuple[pd.Series, int]:
    """Hourly judgements -> a boolean alarm per 10-minute stamp of ``index``.

    Every stamp takes the most recent judgement made at or before it, and only while that
    judgement is less than an hour old. A judgement never reaches stamps before the moment it was
    made: its snapshot's data ends there, so letting it cover earlier stamps would score the model
    on hindsight it did not have. Stamps with no judgement yet (and stamps more than an hour after
    the last one) are not alarmed.

    Matching by "most recent at or before" rather than by exact timestamp matters because the CARE
    event files skip rows: a judgement can be made for a 10-minute stamp the file does not contain,
    and it should still govern the stamps that follow it.

    Returns the series and how many judgements fell outside the event's prediction period
    altogether, which is all CARE-Score evaluates.
    """
    if len(index) == 0:
        raise ValueError("the event's prediction period is empty")
    made = pd.Series(
        [r["pred"] == "anomaly" for r in rows],
        index=pd.to_datetime([r["end_us"] for r in rows], unit="us"),
        dtype="object",
    ).sort_index()
    made = made[~made.index.duplicated(keep="last")]
    within = (made.index >= index[0]) & (made.index <= index[-1])
    made = made[within]
    if made.empty:
        return pd.Series(False, index=index), int((~within).sum())

    # For each stamp: the last judgement at or before it (position -1 means "none yet").
    position = np.searchsorted(made.index.values, index.values, side="right") - 1
    fresh = position >= 0
    age = np.empty(len(index), dtype="timedelta64[ns]")
    age[fresh] = index.values[fresh] - made.index.values[position[fresh]]
    values = np.zeros(len(index), dtype=bool)
    values[fresh] = made.to_numpy(dtype=bool)[position[fresh]] & (age[fresh] < IN_FORCE.to_timedelta64())
    return pd.Series(values, index=index), int((~within).sum())


def persist(rows: list[dict], *, window_hours: int, needed_hours: int) -> list[dict]:
    """Smooth the hourly judgements of ONE event into sustained alarms (PLAN.md 8.1, fix 1).

    An hour alarms when at least ``needed_hours`` of the ``window_hours`` hourly judgements up to
    and including it said anomaly. Scattered pulses are dropped; a run of hits keeps the alarm up
    until the hits age out of the window, which is what CARE's ``criticality > 72`` asks for.
    This is the standard condition-monitoring guard against single-sample chatter -- it changes the
    detector, not the metric.

    Only judgements at or before an hour are counted, so the smoothing cannot use hindsight. Hours
    with less than ``window_hours`` of history count what they have. Call it per event: a window
    must never span two events.

    The returned rows are time ordered copies with ``pred`` replaced and the original kept in
    ``pred_raw``. ``window_hours=needed_hours=1`` is the identity.

    Raises:
        ValueError: If the parameters are not ``1 <= needed_hours <= window_hours``.
    """
    if window_hours < 1 or needed_hours < 1 or needed_hours > window_hours:
        raise ValueError(f"need 1 <= needed_hours <= window_hours, got {needed_hours} and {window_hours}")
    ordered = sorted(rows, key=lambda r: r["end_us"])
    if not ordered:
        return []
    times = np.array([r["end_us"] for r in ordered], dtype=np.int64)
    hits = np.cumsum([r["pred"] == "anomaly" for r in ordered], dtype=np.int64)
    # The window holds the hourly slots t - (window_hours - 1) .. t; allow half a slot of jitter.
    span = int((window_hours - 1) * 3_600_000_000 + 1_800_000_000)
    first = np.searchsorted(times, times - span, side="left")
    in_window = hits - np.where(first > 0, hits[first - 1], 0)
    return [
        row | {"pred": "anomaly" if count >= needed_hours else "normal", "pred_raw": row["pred"]}
        for row, count in zip(ordered, in_window)
    ]


def threshold_rows(rows: list[dict], threshold: float) -> list[dict]:
    """Rows judged ``anomaly`` where ``p_anomaly`` exceeds ``threshold``; the greedy verdict is kept
    as ``pred_greedy``. Pick the threshold on a validation turbine, never on the test turbine."""
    return [
        row | {"pred": "anomaly" if row["p_anomaly"] > threshold else "normal", "pred_greedy": row["pred"]}
        for row in rows
    ]


def rank_auc(scores: list[float], positive: list[bool]) -> float | None:
    """Probability that a random positive scores above a random negative (ties count half).

    None when either class is empty. Mann-Whitney U from average ranks, so no extra dependency.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores_arr, kind="mergesort")
    ranks = np.empty(len(scores_arr), dtype=np.float64)
    sorted_scores = scores_arr[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def calibrated_threshold(healthy_scores: list[float], quantile: float) -> float:
    """The ``quantile`` of a turbine's P(anomaly) on its own healthy period (PLAN.md 8.13).

    Alarming above it makes the turbine alarm on about (1 - quantile) of its healthy hours, whatever
    level the model puts that turbine at -- a per-turbine calibration that reads no label. Uses the
    'higher' interpolation so the threshold is always one of the observed scores.
    """
    if not healthy_scores:
        raise ValueError("no healthy scores to calibrate on")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    return float(np.quantile(np.asarray(healthy_scores, dtype=np.float64), quantile, method="higher"))
