"""Leave-one-turbine-out folds and training-set balancing (PLAN.md 5.5, 5.6)."""

from __future__ import annotations

from collections.abc import Sequence
import random

from care_tslm.config import STEPS_PER_HOUR, TRAIN_NORMAL_STRIDE_HOURS

NORMAL_PER_ANOMALY = 4
"""Target normal:anomaly ratio of a training set (PLAN.md: 3:1 to 5:1)."""


def turbines(meta: Sequence[dict]) -> list[str]:
    return sorted({m["turbine"] for m in meta})


def test_rows(meta: Sequence[dict], test_turbines: set[str]) -> list[dict]:
    """Every prediction snapshot of the test turbines, hourly: the CARE-scored set (true distribution)."""
    return [m for m in meta if m["turbine"] in test_turbines and m["role"] == "prediction"]


def train_rows(
    meta: Sequence[dict],
    test_turbines: set[str],
    *,
    ratio: int = NORMAL_PER_ANOMALY,
    anomaly_stride_hours: int = 1,
    seed: int = 0,
    positive_events: frozenset[str] | None = None,
) -> list[dict]:
    """The balanced training set of a fold.

    - anomaly: every hourly snapshot inside an anomaly event window;
    - normal: training-period snapshots (already 12-hourly) and normal-event prediction snapshots
      thinned to 12-hourly, then down-sampled to ``ratio`` x the anomaly count if there are more.
    Anomaly-event snapshots outside the window (``train_pool`` None) are never used.

    ``positive_events`` (event keys such as ``B7``): only these events' window snapshots are anomaly
    samples; the other anomaly events are left out entirely, like their out-of-window snapshots
    (config.EVIDENCE_EVENTS, PLAN.md 8.13). Normal samples are unaffected, but the ratio cap then
    applies to the smaller anomaly count.
    """
    pool = [m for m in meta if m["turbine"] not in test_turbines and m["train_pool"] is not None]
    a_stride = anomaly_stride_hours * STEPS_PER_HOUR
    anomaly = [m for m in pool if m["train_pool"] == "anomaly" and m["end_step"] % a_stride == 0]
    if positive_events is not None:
        anomaly = [m for m in anomaly if f"{m['farm']}{m['event_id']}" in positive_events]
    stride = TRAIN_NORMAL_STRIDE_HOURS * STEPS_PER_HOUR
    normal = [
        m
        for m in pool
        if m["train_pool"] == "normal" and (m["role"] == "train_normal" or m["end_step"] % stride == 0)
    ]
    rng = random.Random(seed)
    if anomaly and len(normal) > ratio * len(anomaly):
        normal = rng.sample(normal, ratio * len(anomaly))
    rows = anomaly + normal
    rng.shuffle(rows)
    return rows
