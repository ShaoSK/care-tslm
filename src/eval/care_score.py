"""Official CARE-Score for the hourly predictions produced by ``src/train/predict.py``.

Runs in the *separate* energy-fault-detector environment (it pulls TensorFlow, which pins
numpy < 2 and would fight the training env):

    source ~/efd/.venv/bin/activate
    python src/eval/care_score.py --predictions predictions/*.jsonl \
        --care-root ~/data/CARE_To_Compare --out results/care_score_ab.json

What it does, per event:

1. takes the prediction period (the event's *test* dataset) as the 10-minute reference index;
2. expands our hourly judgements onto that index by forward fill, limited to one hour -- a
   judgement made at time t is in force from t until the next one, never before t, so the
   score cannot benefit from hindsight;
3. feeds the boolean alarm series to ``CAREScore.evaluate_event`` together with the official
   ``normal_index`` (``status_type_id == 0``). Wind farm A's status codes are rule-generated
   from the fault date, so for A the normal index is ignored (``ignore_normal_index=True``),
   as the dataset authors intend.

Then it reports the four components (Coverage, Accuracy, Reliability, Earliness), the final
CARE-Score, and a per-event table including the lead time of the first alarm.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.scoring import alarm_series, naive_utc, persist
from energy_fault_detector.evaluation import CAREScore, Care2CompareDataset


def jsonable(value):
    """What the scorer returns that json.dump cannot take: numpy scalars stay numbers,
    timestamps and durations become readable strings."""
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_predictions(patterns: list[str]) -> dict[tuple[str, int], list[dict]]:
    """Group prediction rows by (farm, event_id); later files win on duplicate snapshots."""
    by_event: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    files = [Path(p) for pattern in patterns for p in sorted(glob.glob(pattern))]
    if not files:
        raise SystemExit(f"no prediction files matched {patterns}")
    for path in files:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                by_event[(row["farm"], int(row["event_id"]))][int(row["end_us"])] = row
    return {key: [rows[k] for k in sorted(rows)] for key, rows in by_event.items()}


def care_root(path: Path) -> Path:
    """The directory that holds the ``Wind Farm X`` folders.

    The release unzips to a nested ``CARE_To_Compare/CARE_To_Compare`` on some archives, so accept
    either level -- the same rule as :func:`care_tslm.load_raw.farm_directory`.
    """
    for candidate in (path, path / "CARE_To_Compare"):
        if any((candidate / f"Wind Farm {farm}").is_dir() for farm in "ABC"):
            return candidate
    raise SystemExit(f"no 'Wind Farm X' directory under {path}")


def event_bounds(info: pd.Series, index: pd.Index) -> tuple:
    """Event window start/end, in the same units as ``index``."""
    for start, end in (("event_start", "event_end"), ("event_start_id", "event_end_id")):
        if start in info and end in info:
            lo, hi = info[start], info[end]
            if isinstance(index, pd.DatetimeIndex):
                lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            return lo, hi
    raise KeyError(f"no event bounds in event_info columns {list(info.index)}")


def load_event(dataset, event_id: int, cache: dict | None = None):
    """The event's prediction-period frame on a tz-naive index, loaded once per event."""
    if cache is not None and event_id in cache:
        return cache[event_id]
    test = dataset.load_event_dataset(event_id, test_only=True, index_column="time_stamp").sort_index()
    test.index = naive_utc(pd.DatetimeIndex(test.index))
    if cache is not None:
        cache[event_id] = test
    return test


def evaluate(
    predictions: dict[tuple[str, int], list[dict]],
    dataset,
    *,
    persist_window_hours: int = 1,
    persist_needed_hours: int = 1,
    ignore_normal_index_farms: tuple[str, ...] = ("A",),
    until_end: object = False,
    cache: dict | None = None,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Score every event of ``predictions`` and return (summary, per-event rows).

    Shared by ``main`` and ``src/scripts/tune_persistence.py`` so a parameter sweep scores exactly
    the way a reported run does. ``cache`` lets a sweep read each event's CSV only once.
    """
    care = CAREScore()
    per_event: list[dict] = []
    for (farm, event_id), rows in sorted(predictions.items()):
        info = dataset.get_event_info(event_id)
        test = load_event(dataset, event_id, cache)
        normal_index = test["status_type_id"] == 0
        judgements = persist(rows, window_hours=persist_window_hours, needed_hours=persist_needed_hours)
        alarms, dropped = alarm_series(judgements, test.index)
        start, end = event_bounds(info, test.index)
        gold = str(info["event_label"]).strip().lower()

        metrics = care.evaluate_event(
            event_start=start,
            event_end=end,
            event_label=gold,
            predicted_anomalies=alarms,
            normal_index=normal_index,
            evaluate_until_event_end=until_end,
            event_id=event_id,
            ignore_normal_index=farm in ignore_normal_index_farms,
        )

        fired = alarms[alarms]
        lead_days = None
        if gold == "anomaly" and len(fired):
            lead_days = round((pd.Timestamp(end) - fired.index[0]).total_seconds() / 86400, 2)
        per_event.append(
            dict(metrics)
            | {
                "farm": farm,
                "turbine": rows[0].get("turbine"),
                "hourly_predictions": len(rows),
                "predictions_outside_prediction_period": dropped,
                "alarm_fraction": round(float(alarms.mean()), 4),
                "raw_anomaly_hours": sum(r["pred_raw"] == "anomaly" for r in judgements),
                "persisted_anomaly_hours": sum(r["pred"] == "anomaly" for r in judgements),
                "first_alarm_lead_days": lead_days,
            }
        )
        if verbose:
            print(
                f"{farm}{event_id:<4} {gold:<7} acc={metrics['accuracy']:.3f} "
                f"crit={metrics['max_criticality']:.0f} alarms={alarms.mean():.1%} "
                f"lead={lead_days if lead_days is not None else '-'}d"
                + (f"  [{dropped} preds outside]" if dropped else "")
            )

    summary = {
        "n_events": len(per_event),
        "persist_window_hours": persist_window_hours,
        "persist_needed_hours": persist_needed_hours,
        "care_score": care.get_final_score(),
        "coverage": care.calculate_avg_coverage(),
        "accuracy": care.calculate_avg_accuracy(),
        "reliability": care.calculate_reliability(),
        "earliness": care.calculate_avg_earliness(),
    }
    return summary, per_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True, help="jsonl files from predict.py (globs ok)")
    parser.add_argument("--care-root", type=Path, required=True, help="unzipped CARE_To_Compare directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ignore-normal-index-farms",
        nargs="*",
        default=["A"],
        help="farms whose status codes are rule-generated (default: A)",
    )
    parser.add_argument("--evaluate-until-event-end", default="False", choices=["False", "True", "normal_only", "anomaly_only"])
    parser.add_argument(
        "--persist-window-hours",
        type=int,
        default=1,
        help="alarm persistence (PLAN.md 8.1 fix 1): the length of the look-back window; 1 disables it",
    )
    parser.add_argument(
        "--persist-needed-hours",
        type=int,
        default=1,
        help="how many hours of the window must say anomaly before the alarm goes up",
    )
    args = parser.parse_args()

    until_end: object = {"False": False, "True": True}.get(
        args.evaluate_until_event_end, args.evaluate_until_event_end
    )
    predictions = load_predictions(args.predictions)
    dataset = Care2CompareDataset(path=care_root(args.care_root))
    summary, per_event = evaluate(
        predictions,
        dataset,
        persist_window_hours=args.persist_window_hours,
        persist_needed_hours=args.persist_needed_hours,
        ignore_normal_index_farms=tuple(args.ignore_normal_index_farms),
        until_end=until_end,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "events": per_event}, indent=2, default=jsonable))
    print("\n=== CARE-Score ===")
    for k, v in summary.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
