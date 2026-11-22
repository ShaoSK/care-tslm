"""End-to-end checks of the preprocessing on the synthetic farm: shapes, leakage rules, signal, text."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import folds, text  # noqa: E402
from care_tslm.build import ArrayCache, build_store, load_events, load_meta  # noqa: E402
from care_tslm.config import LONG_POINTS, LONG_STEPS, SHORT_POINTS  # noqa: E402
from care_tslm.load_raw import load_farm  # noqa: E402
from tests.synthetic import write_farm  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("care")
    write_farm(root)
    out = root / "store"
    build_store(load_farm(root, "A"), out, log=lambda *_: None)
    return out


def test_shapes_and_counts(store: Path) -> None:
    meta, events = load_meta(store), load_events(store)
    assert len(events) == 4
    long, short = ArrayCache(store).get(meta[0])
    assert long.shape == (13, LONG_POINTS) and short.shape == (13, SHORT_POINTS)
    pred = [m for m in meta if m["role"] == "prediction" and m["event_id"] == 3]
    assert len(pred) == 20 * 24  # hourly over the 20-day prediction period


def test_no_snapshot_reaches_the_baseline(store: Path) -> None:
    meta, events = load_meta(store), load_events(store)
    for m in meta:
        baseline_end = events[f"A{m['event_id']}"]["baseline_end"]
        assert m["end_step"] - LONG_STEPS + 1 >= baseline_end


def test_labels_and_pools(store: Path) -> None:
    meta = load_meta(store)
    for m in meta:
        if m["role"] == "train_normal":
            assert m["target"] == "normal" and m["train_pool"] == "normal"
        if m["event_label"] == "anomaly" and m["role"] == "prediction":
            assert m["train_pool"] == ("anomaly" if m["in_window"] else None)
    assert sum(m["target"] == "anomaly" for m in meta) == 2 * 7 * 24


def test_folds_do_not_mix_turbines(store: Path) -> None:
    meta = load_meta(store)
    train = folds.train_rows(meta, {"A-0"})
    assert all(m["turbine"] != "A-0" for m in train)
    assert {m["target"] for m in train} == {"anomaly", "normal"}
    n_anomaly = sum(m["target"] == "anomaly" for m in train)
    assert sum(m["target"] == "normal" for m in train) <= folds.NORMAL_PER_ANOMALY * n_anomaly
    assert all(m["turbine"] == "A-0" for m in folds.test_rows(meta, {"A-0"}))


def test_injected_fault_is_visible(store: Path) -> None:
    meta = load_meta(store)
    last = max((m for m in meta if m["event_id"] == 1), key=lambda m: m["end_step"])
    normal = [m["components"]["sensor_12"]["mean_7d"] for m in meta if m["train_pool"] == "normal"]
    assert last["components"]["sensor_12"]["mean_7d"] > 1.5
    assert np.percentile(np.abs(normal), 99) < 1.0


def test_rationale_names_the_faulty_group(store: Path) -> None:
    meta, events = load_meta(store), load_events(store)
    thresholds = text.fit_thresholds(m for m in meta if m["turbine"] != "A-0")
    last = max((m for m in meta if m["event_id"] == 1), key=lambda m: m["end_step"])
    answer, facts = text.rationale(last, events["A1"]["channel_text"], thresholds)
    assert answer.endswith("Answer: anomaly")
    assert facts["deviating_groups"][0] == "gearbox"
    assert "gearbox" in answer


def test_channel_prompts(store: Path) -> None:
    meta, events = load_meta(store), load_events(store)
    long, short = ArrayCache(store).get(meta[-1])
    prompts = text.channel_prompts("A", long, short, events["A4"]["channel_text"])
    assert len(prompts) == 26
    assert all(np.isfinite(s).all() for _, s in prompts)
    assert {len(s) for _, s in prompts} == {LONG_POINTS, SHORT_POINTS}
