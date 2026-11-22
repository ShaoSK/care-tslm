"""``CAREWindFaultCoTDataset``: our snapshot store as an OpenTSLM ``QADataset`` (PLAN.md 6-7).

Each sample follows OpenTSLM's CoT format (as HAR-CoT / ECG-QA-CoT):

    pre_prompt   task description, possible answers, "end with Answer: "
    series       26 (farm A) or 32 (farm B) TextTimeSeriesPrompts: every channel at 30 d / 4 h and
                 7.5 d / 1 h, 180 points each
    post_prompt  "Rationale:"
    answer       rule-based rationale ending in "Answer: anomaly|normal" (+ EOS, added by QADataset)

A fold is defined by its test turbines. QADataset caches the formatted splits on the class, so
:func:`make_dataset_class` creates one class per fold; ``CurriculumTrainer`` then calls
``cls("train" | "validation" | "test", EOS_TOKEN=...)`` as for its own stages.

The sample building (:func:`build_split_rows`) needs only numpy, so it is tested without OpenTSLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re

from care_tslm import folds, text
from care_tslm.build import ArrayCache, load_events, load_meta
import numpy as np

from care_tslm.config import EVIDENCE_EVENTS, PAIR_ROLE, STEPS_PER_HOUR, channel_list


@dataclass(frozen=True)
class FoldSpec:
    """Which data a fold trains, validates and tests on."""

    store: Path
    test_turbines: tuple[str, ...]
    val_turbine: str | None = "auto"
    """A training turbine held out for early stopping ("auto": the first one after the test turbines
    that has an anomaly event, in the same farm if possible). None: a random 10 % of training rows,
    which is faster but optimistic (neighbouring hours are near-duplicates)."""
    ratio: int = folds.NORMAL_PER_ANOMALY
    anomaly_stride_hours: int = 1
    """>1 thins the hourly anomaly snapshots of the training set (saves GPU time)."""
    legacy_text: bool = False
    """Use the 11-18 prompt and answer text (:data:`text.PRE_PROMPT_LEGACY` + :func:`text.rationale`):
    prose first, verdict last, no Component or Evidence field.

    This is the configuration the v1 run was trained on -- the only one that beat the residual
    baseline (PLAN.md 8.5). The three-field rewrite replaced it in place, so without this switch that
    result cannot be retrained."""
    answer_format: str = "fields"
    """Order of the answer fields (:data:`text.FIELD_ORDER`): ``fields`` (verdict first), ``prefix``
    (a fixed one-sentence assessment before the verdict) or ``last`` (verdict after the evidence).
    Only for the field-based answer, so it cannot be combined with ``legacy_text`` or ``answer_only``."""
    answer_only: bool = False
    """Train on ``Answer: anomaly|normal`` alone, with no Component or Evidence field and a prompt
    that asks for nothing else. The whole loss then falls on the verdict -- the pure-classification
    reference point this project never had."""
    test_stride_hours: int = 1
    """>1 thins the test set. CARE-Score needs 1 (every hour); use more for quick checks."""
    evidence_only: bool = False
    """Train and validate on the anomaly windows of :data:`config.EVIDENCE_EVENTS` only (PLAN.md 8.13):
    the other anomaly events show nothing in our channels, so their windows teach shortcuts. The
    test split is unchanged."""
    offset_aug: float = 0.0
    """>0: training samples get a random constant, uniform in +-offset_aug sigma, added to each
    component residual (PLAN.md 8.13). Fixed per sample (seeded by the snapshot), never on
    validation or test."""
    calibration: bool = False
    """The test split becomes the test turbines' TRAINING-period snapshots (role ``train_normal``,
    the second half of each training year): healthy by CARE's definition, never trained on, scored
    to set that turbine's own P(anomaly) threshold (PLAN.md 8.13). Not for CARE-Score."""
    calibration_every: int = 1
    """Keep every n-th calibration snapshot (they are 12-hourly)."""
    max_val: int = 400
    max_train: int | None = None
    """Cap for smoke tests (random subset, keeps the class mix roughly)."""
    max_test: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.answer_format not in text.FIELD_ORDER:
            raise ValueError(f"answer_format {self.answer_format!r} is not one of {sorted(text.FIELD_ORDER)}")
        if self.answer_format != "fields" and (self.legacy_text or self.answer_only):
            raise ValueError("answer_format only applies to the field-based answer, not legacy_text or answer_only")
        if self.legacy_text and self.answer_only:
            raise ValueError("legacy_text and answer_only are exclusive")

    @property
    def name(self) -> str:
        return "test_" + "_".join(self.test_turbines)


def _pick_val_turbine(meta: list[dict], spec: FoldSpec) -> str | None:
    if spec.val_turbine != "auto":
        return spec.val_turbine
    anomalous = {
        m["turbine"]
        for m in meta
        if m["event_label"] == "anomaly"
        and (not spec.evidence_only or f"{m['farm']}{m['event_id']}" in EVIDENCE_EVENTS)
    }
    with_anomaly = sorted(anomalous - set(spec.test_turbines))
    farms = {t.split("-")[0] for t in spec.test_turbines}
    same_farm = [t for t in with_anomaly if t.split("-")[0] in farms]
    if spec.evidence_only and len(same_farm) < 2:
        # Holding out the farm's last positive turbine would leave training no positive of that farm.
        same_farm = [t for t in with_anomaly if t not in same_farm]
    candidates = same_farm or with_anomaly
    if not candidates:
        return None
    # Deterministic but different per fold: the first candidate after the test turbine in sort order.
    after = [t for t in candidates if t > max(spec.test_turbines)]
    return (after or candidates)[0]


def store_has_pairs(events: dict) -> bool:
    """Whether the store was built with ``--with-pairs`` (pair-difference channels at the end)."""
    return any(PAIR_ROLE in info["roles"] for info in events.values())


def sample_offsets(meta: dict, roles: list[str], amplitude: float, seed: int) -> np.ndarray:
    """One random offset per channel (0 for non-component channels), fixed by the snapshot."""
    rng = np.random.default_rng([seed, int(meta["event_id"]), int(meta["end_step"]), ord(meta["farm"])])
    draws = rng.uniform(-amplitude, amplitude, size=len(roles))
    return np.where(np.asarray(roles) == "component", draws, 0.0)


def store_has_raw(events: dict) -> bool:
    """Whether the store was built with ``--with-raw`` (every component channel carried twice)."""
    return any("component_raw" in info["roles"] for info in events.values())


def _row(
    meta: dict,
    events: dict,
    cache: ArrayCache,
    thresholds: text.Thresholds,
    *,
    with_raw: bool = False,
    with_pairs: bool = False,
    offset_aug: float = 0.0,
    seed: int = 0,
    signal_thresholds: text.Thresholds | None = None,
    answer_only: bool = False,
    legacy_text: bool = False,
    answer_format: str = "fields",
) -> dict:
    info = events[f"{meta['farm']}{meta['event_id']}"]
    long, short = cache.get(meta)
    offsets = sample_offsets(meta, info["roles"], offset_aug, seed) if offset_aug > 0 else None
    series = text.channel_prompts(
        meta["farm"], long, short, info["channel_text"], with_raw=with_raw, with_pairs=with_pairs, offsets=offsets
    )
    keys = {
        "post_prompt": text.POST_PROMPT,
        "series_text": [t for t, _ in series],
        "series": [s for _, s in series],
        "label": meta["target"],
        "farm": meta["farm"],
        "event_id": meta["event_id"],
        "turbine": meta["turbine"],
        "end_step": meta["end_step"],
        "end_us": meta["end_us"],
        "status_end": meta["status_end"],
    }
    if legacy_text:
        answer, facts = text.rationale(meta, info["channel_text"], thresholds)
        return keys | {
            "pre_prompt": text.PRE_PROMPT_LEGACY,
            "rationale": answer,
            "fields": {"answer": meta["target"]},
            "consistent": facts["consistent"],
        }
    if answer_only:
        return keys | {
            "pre_prompt": text.PRE_PROMPT_ANSWER_ONLY,
            "rationale": f"Answer: {meta['target']}",
            "fields": {"answer": meta["target"]},
            "consistent": True,
        }
    fields, facts = text.rationale_fields(
        meta,
        info["channel_text"],
        thresholds,
        info.get("baseline"),
        info.get("fault_component"),
        signal_thresholds,
        info.get("pair_baseline"),
        info.get("load_baseline"),
    )
    return keys | {
        "pre_prompt": text.pre_prompt(answer_format)
        + (text.RAW_PROMPT_NOTE if with_raw else "")
        + (text.PAIR_PROMPT_NOTE if with_pairs else ""),
        "rationale": text.render(fields, answer_format),
        "fields": fields,
        "consistent": facts["consistent"],
    }


def build_split_rows(spec: FoldSpec) -> tuple[list[dict], list[dict], list[dict], dict]:
    """(train, validation, test) rows of a fold, and a summary of what went where."""
    meta = load_meta(spec.store)
    events = load_events(spec.store)
    test_set = set(spec.test_turbines)
    unknown = test_set - set(folds.turbines(meta))
    if unknown:
        raise ValueError(f"test turbines {sorted(unknown)} are not in the store")

    val_turbine = _pick_val_turbine(meta, spec)
    held_out = test_set | ({val_turbine} if val_turbine else set())
    positives = EVIDENCE_EVENTS if spec.evidence_only else None
    train_meta = folds.train_rows(
        meta,
        held_out,
        ratio=spec.ratio,
        anomaly_stride_hours=spec.anomaly_stride_hours,
        seed=spec.seed,
        positive_events=positives,
    )
    rng = random.Random(spec.seed)
    if val_turbine:
        others = set(folds.turbines(meta)) - {val_turbine}
        val_meta = folds.train_rows(meta, others, ratio=spec.ratio, seed=spec.seed, positive_events=positives)
    else:
        rng.shuffle(train_meta)
        cut = max(1, len(train_meta) // 10)
        val_meta, train_meta = train_meta[:cut], train_meta[cut:]
    if len(val_meta) > spec.max_val:
        val_meta = rng.sample(val_meta, spec.max_val)

    stride = spec.test_stride_hours * STEPS_PER_HOUR
    if spec.calibration:
        test_meta = [m for m in meta if m["turbine"] in test_set and m["role"] == "train_normal"]
        test_meta = test_meta[:: spec.calibration_every]
    else:
        test_meta = [m for m in folds.test_rows(meta, test_set) if m["end_step"] % stride == 0]
    if spec.max_train is not None and len(train_meta) > spec.max_train:
        train_meta = rng.sample(train_meta, spec.max_train)
    if spec.max_test is not None and len(test_meta) > spec.max_test:
        test_meta = sorted(rng.sample(test_meta, spec.max_test), key=lambda m: (m["event_id"], m["end_step"]))

    # Thresholds for the rationale: training-side turbines only (never the test turbines).
    thresholds = text.fit_thresholds(m for m in meta if m["turbine"] not in test_set)
    baselines = {key: info.get("baseline") or {} for key, info in events.items()}
    signal_thresholds = text.fit_signal_thresholds(
        (m for m in meta if m["turbine"] not in test_set), baselines
    )
    cache = ArrayCache(spec.store)
    with_raw = store_has_raw(events)
    with_pairs = store_has_pairs(events)
    splits = tuple(
        [
            _row(
                m, events, cache, thresholds,
                with_raw=with_raw, with_pairs=with_pairs, offset_aug=aug, seed=spec.seed,
                signal_thresholds=signal_thresholds, answer_only=spec.answer_only,
                legacy_text=spec.legacy_text, answer_format=spec.answer_format,
            )
            for m in rows
        ]
        for rows, aug in ((train_meta, spec.offset_aug), (val_meta, 0.0), (test_meta, 0.0))
    )

    def counts(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "anomaly": sum(r["label"] == "anomaly" for r in rows),
            "normal": sum(r["label"] == "normal" for r in rows),
            "rationale_consistent": round(sum(r["consistent"] for r in rows) / max(1, len(rows)), 3),
        }

    summary = {
        "fold": spec.name,
        "test_turbines": list(spec.test_turbines),
        "val_turbine": val_turbine,
        "with_raw": with_raw,
        "with_pairs": with_pairs,
        "evidence_only": spec.evidence_only,
        "offset_aug": spec.offset_aug,
        "calibration": spec.calibration,
        "channels_per_farm": {
            farm: len(channel_list(farm, with_raw=with_raw, with_pairs=with_pairs))
            for farm in sorted({m["farm"] for m in meta})
        },
        "train": counts(splits[0]),
        "validation": counts(splits[1]),
        "test": counts(splits[2]),
    }
    return splits[0], splits[1], splits[2], summary


ANSWER = re.compile(r"Answer:\s*(anomaly|normal)\b", re.IGNORECASE)


def parse_answer(generated: str) -> str | None:
    """The label of a generated rationale: the last 'Answer: x', or None if there is none."""
    found = ANSWER.findall(generated)
    return found[-1].lower() if found else None


def make_dataset_class(spec: FoldSpec):
    """A ``QADataset`` subclass bound to one fold. Requires OpenTSLM on the path."""
    from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
    from opentslm.time_series_datasets.QADataset import QADataset

    train, val, test, summary = build_split_rows(spec)

    class CAREWindFaultCoTDataset(QADataset):
        fold = spec
        fold_summary = summary
        train_farms = [row["farm"] for row in train]
        """The farm of each training sample, in dataset order (for :class:`SameFarmBatchSampler`)."""

        def _load_splits(self):
            return train, val, test

        def _get_answer(self, row) -> str:
            return row["rationale"]

        def _get_pre_prompt(self, row) -> str:
            return row["pre_prompt"]

        def _get_post_prompt(self, row) -> str:
            return row["post_prompt"]

        def _get_text_time_series_prompt_list(self, row):
            return [TextTimeSeriesPrompt(t, s) for t, s in zip(row["series_text"], row["series"], strict=True)]

        @staticmethod
        def get_labels() -> list[str]:
            return ["anomaly", "normal"]

        def _format_sample(self, row):
            sample = super()._format_sample(row)
            for key in ("label", "farm", "event_id", "turbine", "end_step", "end_us", "status_end"):
                sample[key] = row[key]
            return sample

    CAREWindFaultCoTDataset.__name__ = f"CAREWindFaultCoTDataset_{spec.name}"
    return CAREWindFaultCoTDataset
