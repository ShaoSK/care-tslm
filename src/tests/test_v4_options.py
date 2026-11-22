"""The v4 options of PLAN.md 8.13: pair-difference channels, evidence-only positives, the random
offset augmentation and the per-turbine calibration split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import folds, opentslm_dataset, text  # noqa: E402
from care_tslm.build import ArrayCache, build_store, load_events, load_meta  # noqa: E402
from care_tslm.config import CHANNELS, PAIR_ROLE, channel_list, sensor_pairs  # noqa: E402
from care_tslm.load_raw import load_farm  # noqa: E402
from care_tslm.opentslm_dataset import FoldSpec, build_split_rows, sample_offsets  # noqa: E402
from care_tslm.scoring import calibrated_threshold  # noqa: E402
from tests.synthetic import write_farm  # noqa: E402

N_A = len(CHANNELS["A"])


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("care_v4")
    write_farm(root)
    plain, pairs = root / "plain", root / "pairs"
    build_store(load_farm(root, "A"), plain, log=lambda *_: None)
    build_store(load_farm(root, "A"), pairs, with_pairs=True, log=lambda *_: None)
    return plain, pairs


# --- pair channels -------------------------------------------------------------------------------


def test_channel_list_appends_one_channel_per_sensor_pair():
    with_pairs = channel_list("A", with_pairs=True)
    assert list(with_pairs[:N_A]) == list(CHANNELS["A"])
    added = with_pairs[N_A:]
    assert [c.sensor for c in added] == [f"{a}-{b}" for _, a, b in sensor_pairs("A")]
    assert all(c.role == PAIR_ROLE for c in added)
    assert len(channel_list("B", with_pairs=True)) == len(CHANNELS["B"]) + len(sensor_pairs("B"))
    # Raw copies (if any) come before the pairs, so the residual channels never move.
    both = channel_list("A", with_raw=True, with_pairs=True)
    assert [c.role for c in both[-len(added) :]] == [PAIR_ROLE] * len(added)


def test_pair_store_leaves_every_other_channel_and_the_labels_untouched(stores):
    plain, pairs = stores
    meta_plain, meta_pairs = load_meta(plain), load_meta(pairs)
    assert meta_plain == meta_pairs, "labels, pools and statistics must not depend on the extra channels"
    for m in (meta_plain[0], meta_plain[-1]):
        long_plain, short_plain = ArrayCache(plain).get(m)
        long_pairs, short_pairs = ArrayCache(pairs).get(m)
        assert long_pairs.shape[0] == N_A + len(sensor_pairs("A"))
        for a, b in ((long_plain, long_pairs[:N_A]), (short_plain, short_pairs[:N_A])):
            assert np.array_equal(np.nan_to_num(a, nan=99), np.nan_to_num(b, nan=99))


def test_pair_channel_is_the_centred_scaled_residual_difference(stores):
    plain, pairs = stores
    events = load_events(pairs)
    m = next(x for x in load_meta(pairs) if x["event_id"] == 3 and x["role"] == "prediction")
    info = events["A3"]
    names = info["channels"]
    long_plain, _ = ArrayCache(plain).get(m)
    long_pairs, _ = ArrayCache(pairs).get(m)
    stats = info["pair_baseline"]["sensor_13-sensor_14"]
    a, b = names.index("sensor_13"), names.index("sensor_14")
    expected = (long_plain[a].astype(np.float64) - long_plain[b] - stats["mean"]) / stats["std"]
    got = long_pairs[names.index("sensor_13-sensor_14")].astype(np.float64)
    finite = np.isfinite(expected) & np.isfinite(got)
    assert finite.mean() > 0.8
    # Bin mean of a difference = difference of bin means where both sensors share the gaps.
    assert np.allclose(got[finite], expected[finite], atol=0.05)


def test_a_one_sided_fault_moves_the_pair_channel_a_healthy_period_does_not(stores):
    _, pairs = stores
    events = load_events(pairs)
    i = events["A2"]["channels"].index("sensor_13-sensor_14")  # A2: generator bearing fault on sensor_13 only
    cache = ArrayCache(pairs)
    rows = [m for m in load_meta(pairs) if m["event_id"] == 2]
    healthy = np.nanmean([np.nanmean(cache.get(m)[1][i]) for m in rows if m["role"] == "train_normal"])
    # The synthetic fault is a ramp to +4 degC on sensor_13 alone: look at the last day before it.
    last = max((m for m in rows if m["in_window"]), key=lambda m: m["end_step"])
    faulty = np.nanmean(cache.get(last)[1][i][-24:].astype(np.float64))
    assert abs(healthy) < 0.5
    assert faulty > healthy + 1.5


def test_pair_prompts_name_both_sensors_and_the_prompt_explains_them(stores):
    _, pairs = stores
    _, _, test, summary = build_split_rows(FoldSpec(pairs, ("A-0",), val_turbine=None, max_test=3))
    assert summary["with_pairs"] and summary["channels_per_farm"]["A"] == N_A + len(sensor_pairs("A"))
    row = test[0]
    assert len(row["series"]) == 2 * (N_A + len(sensor_pairs("A")))
    assert any(" minus " in t and "difference of residuals" in t for t in row["series_text"])
    assert text.PAIR_PROMPT_NOTE in row["pre_prompt"]


# --- evidence-only positives ---------------------------------------------------------------------


def test_positive_events_keep_only_those_windows_as_anomaly(stores):
    plain, _ = stores
    meta = load_meta(plain)
    every = folds.train_rows(meta, set())
    only = folds.train_rows(meta, set(), positive_events=frozenset({"A2"}))
    assert {f"A{m['event_id']}" for m in every if m["train_pool"] == "anomaly"} == {"A1", "A2"}
    assert {f"A{m['event_id']}" for m in only if m["train_pool"] == "anomaly"} == {"A2"}
    assert all(m["train_pool"] == "normal" for m in only if m["event_id"] == 1), "A1's window is dropped, not relabelled"


def test_evidence_only_fold_trains_on_the_evidence_events_and_tests_on_everything(stores, monkeypatch):
    plain, _ = stores
    monkeypatch.setattr(opentslm_dataset, "EVIDENCE_EVENTS", frozenset({"A1"}))
    base = build_split_rows(FoldSpec(plain, ("A-3",), val_turbine=None))
    evid = build_split_rows(FoldSpec(plain, ("A-3",), val_turbine=None, evidence_only=True))
    pos = lambda rows: {r["event_id"] for r in rows if r["label"] == "anomaly"}  # noqa: E731
    assert pos(base[0] + base[1]) == {1, 2}
    assert pos(evid[0] + evid[1]) == {1}
    assert [r["end_step"] for r in evid[2]] == [r["end_step"] for r in base[2]], "test split unchanged"


def test_evidence_only_never_holds_out_a_farms_last_positive_turbine():
    meta = [
        {"turbine": t, "farm": t[0], "event_id": e, "event_label": "anomaly"}
        for t, e in (("A-0", 0), ("A-13", 45), ("A-21", 22), ("B-11", 19), ("B-13", 7), ("B-14", 34))
    ]
    spec = FoldSpec(Path("."), ("A-13",), evidence_only=True)
    # A-0 holds the only other A-farm evidence event: validating on it would leave training no A positive.
    assert opentslm_dataset._pick_val_turbine(meta, spec) == "B-11"
    assert opentslm_dataset._pick_val_turbine(meta, FoldSpec(Path("."), ("A-13",))) == "A-21"
    assert opentslm_dataset._pick_val_turbine(meta, FoldSpec(Path("."), ("B-11",), evidence_only=True)) == "B-13"


# --- offset augmentation -------------------------------------------------------------------------


def test_offsets_touch_component_channels_only_and_are_fixed_per_snapshot():
    roles = [c.role for c in channel_list("A", with_pairs=True)]
    m = {"event_id": 7, "end_step": 1234, "farm": "A"}
    a, b = sample_offsets(m, roles, 1.0, 0), sample_offsets(m, roles, 1.0, 0)
    assert np.array_equal(a, b)
    comp = np.asarray(roles) == "component"
    assert np.all(a[~comp] == 0) and np.all(np.abs(a[comp]) <= 1.0) and np.any(a[comp] != 0)
    assert not np.array_equal(a, sample_offsets(m | {"end_step": 1235}, roles, 1.0, 0))


def test_augmentation_shifts_training_curves_only(stores):
    _, pairs = stores
    plain_rows = build_split_rows(FoldSpec(pairs, ("A-0",), val_turbine="A-1", max_test=20))
    aug_rows = build_split_rows(FoldSpec(pairs, ("A-0",), val_turbine="A-1", max_test=20, offset_aug=1.0))
    roles = [c.role for c in channel_list("A", with_pairs=True)] * 2
    for split in (1, 2):  # validation and test: identical
        assert [r["series_text"] for r in plain_rows[split]] == [r["series_text"] for r in aug_rows[split]]
    r0, r1 = plain_rows[0][0], aug_rows[0][0]
    assert r0["rationale"] == r1["rationale"], "the target text is not changed by the augmentation"
    moved = [not np.allclose(x, y) for x, y in zip(r0["series"], r1["series"], strict=True)]
    for role, m in zip(roles, moved, strict=True):
        if role != "component":
            assert not m, f"{role} channel must not be shifted"
    assert any(m for role, m in zip(roles, moved, strict=True) if role == "component")


# --- calibration ---------------------------------------------------------------------------------


def test_calibration_split_is_the_test_turbines_training_period(stores):
    plain, _ = stores
    meta = load_meta(plain)
    _, _, calib, summary = build_split_rows(FoldSpec(plain, ("A-0",), val_turbine=None, calibration=True))
    expected = [m["end_step"] for m in meta if m["turbine"] == "A-0" and m["role"] == "train_normal"]
    assert [r["end_step"] for r in calib] == expected and summary["calibration"]
    assert {r["label"] for r in calib} == {"normal"}
    pred_first = min(m["end_step"] for m in meta if m["turbine"] == "A-0" and m["role"] == "prediction")
    assert max(r["end_step"] for r in calib) < pred_first, "never reaches into the scored period"
    _, _, thinned, _ = build_split_rows(
        FoldSpec(plain, ("A-0",), val_turbine=None, calibration=True, calibration_every=2)
    )
    assert [r["end_step"] for r in thinned] == expected[::2]


def test_calibrated_threshold_is_an_observed_quantile():
    scores = [i / 100 for i in range(100)]
    t = calibrated_threshold(scores, 0.95)
    assert t in scores and abs(t - 0.95) < 0.011
    assert sum(s > t for s in scores) / len(scores) <= 0.05
    with pytest.raises(ValueError):
        calibrated_threshold([], 0.95)
    with pytest.raises(ValueError):
        calibrated_threshold(scores, 1.0)
