"""Pick the P(anomaly) threshold on the validation turbine, then score the test turbine with it.

Runs in the energy-fault-detector environment on the output of ``src/train/score_answers.py``:

    source ~/efd/.venv/bin/activate
    python src/scripts/tune_threshold.py --val ~/predictions/scored_A-10_ao.jsonl \
        --test ~/predictions/scored_A-0_ao.jsonl --care-root ~/data/CARE_To_Compare

The candidate thresholds are the validation scores' quantiles that make the validation turbine alarm
on 0-60 % of its hours, plus 0.5 (what greedy decoding does). Alarm persistence stays at the fixed
(24, 1) of PLAN.md 8.2 so that only the threshold is chosen here.

NO-LEAKAGE RULE (PLAN.md 5.1): the threshold is chosen on ``--val`` alone. ``--test`` is scored once,
with that threshold; its ROC AUC is printed as a diagnostic of how well the scores rank, not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.scoring import rank_auc, threshold_rows  # noqa: E402
from eval.care_score import care_root, evaluate, jsonable, load_predictions  # noqa: E402

from energy_fault_detector.evaluation import Care2CompareDataset  # noqa: E402

ALARM_RATES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


def thresholded(predictions: dict, threshold: float) -> dict:
    return {key: threshold_rows(rows, threshold) for key, rows in predictions.items()}


def alarm_rate(predictions: dict, threshold: float) -> float:
    rows = [r for rs in predictions.values() for r in rs]
    return sum(r["p_anomaly"] > threshold for r in rows) / len(rows)


def score(predictions: dict, dataset, threshold: float, cache: dict, window: int, needed: int):
    return evaluate(
        thresholded(predictions, threshold),
        dataset,
        persist_window_hours=window,
        persist_needed_hours=needed,
        cache=cache,
        verbose=False,
    )


def diagnostics(predictions: dict, event_labels: dict) -> dict:
    """How well P(anomaly) ranks the test snapshots, by snapshot label and by event label."""
    rows = [(key, r) for key, rs in predictions.items() for r in rs]
    p = [r["p_anomaly"] for _, r in rows]
    return {
        "auc_snapshot_label": rank_auc(p, [r["label"] == "anomaly" for _, r in rows]),
        "auc_event_label": rank_auc(p, [event_labels[key] == "anomaly" for key, _ in rows]),
        "mean_p_by_event": {
            f"{k[0]}{k[1]} ({event_labels[k]})": float(np.mean([r["p_anomaly"] for r in rs]))
            for k, rs in sorted(predictions.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", nargs="+", required=True, help="scored rows of the validation turbine")
    parser.add_argument("--test", nargs="+", required=True, help="scored rows of the test turbine")
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--persist-window-hours", type=int, default=24)
    parser.add_argument("--persist-needed-hours", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = Care2CompareDataset(path=care_root(args.care_root))
    window, needed = args.persist_window_hours, args.persist_needed_hours
    val, test = load_predictions(args.val), load_predictions(args.test)
    cache: dict = {}

    val_p = np.array([r["p_anomaly"] for rs in val.values() for r in rs])
    grid = sorted({0.5} | {float(np.quantile(val_p, 1 - rate)) for rate in ALARM_RATES if rate > 0} | {1.0})
    print(f"validation: {len(val)} events, {len(val_p)} snapshots; persistence ({window},{needed})\n")
    print(f"{'threshold':>10} {'val alarms':>10} {'CARE':>7} {'cov':>6} {'acc':>6} {'rel':>6} {'early':>6}")
    sweep = []
    for t in grid:
        summary, _ = score(val, dataset, t, cache, window, needed)
        rate = alarm_rate(val, t)
        sweep.append(summary | {"threshold": t, "val_alarm_rate": rate})
        print(
            f"{t:>10.4g} {rate:>10.1%} {summary['care_score']:>7.4f} {summary['coverage']:>6.3f} "
            f"{summary['accuracy']:>6.3f} {summary['reliability']:>6.3f} {summary['earliness']:>6.3f}"
            + ("   <- greedy" if t == 0.5 else "")
        )
    # Best CARE; on a tie the higher threshold (fewer alarms).
    best = max(sweep, key=lambda s: (s["care_score"], s["threshold"]))
    t = best["threshold"]
    print(f"\nchosen on validation: threshold {t:.4g} (val CARE {best['care_score']:.4f})\n")

    summary, per_event = score(test, dataset, t, cache, window, needed)
    greedy, _ = score(test, dataset, 0.5, cache, window, needed)
    event_labels = {(e["farm"], int(e["event_id"])): e["event_label"] for e in per_event}
    diag = diagnostics(test, event_labels)
    test_rows = [r for rs in test.values() for r in rs]
    print("=== test turbine ===")
    print(f"raw alarms at chosen threshold: {sum(r['p_anomaly'] > t for r in test_rows)} / {len(test_rows)}")
    print(f"raw alarms greedy (0.5):        {sum(r['p_anomaly'] > 0.5 for r in test_rows)} / {len(test_rows)}")
    print(f"CARE at chosen threshold: {summary['care_score']:.4f}  (cov {summary['coverage']:.3f} "
          f"acc {summary['accuracy']:.3f} rel {summary['reliability']:.3f} early {summary['earliness']:.3f})")
    print(f"CARE greedy (0.5):        {greedy['care_score']:.4f}")
    for e in per_event:
        print(f"   {e['farm']}{e['event_id']:<4} {e['event_label']:8} crit={e['max_criticality']:.0f}")
    print("\ndiagnostic only (not used to choose anything):")
    auc_s, auc_e = diag["auc_snapshot_label"], diag["auc_event_label"]
    print(f"   ROC AUC of P(anomaly), snapshot label: {auc_s:.3f}" if auc_s is not None else "   AUC n/a")
    print(f"   ROC AUC of P(anomaly), event label:    {auc_e:.3f}" if auc_e is not None else "   AUC n/a")
    for name, mean_p in diag["mean_p_by_event"].items():
        print(f"   mean P(anomaly) {name}: {mean_p:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"sweep": sweep, "threshold": t, "test": summary, "test_greedy": greedy,
                 "test_events": per_event, "diagnostics": diag},
                indent=2,
                default=jsonable,
            )
        )
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
