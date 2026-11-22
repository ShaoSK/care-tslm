"""Pick the alarm-persistence parameters N, k on a turbine the model was NOT trained on.

Runs in the energy-fault-detector environment and scores with the very same ``evaluate`` the
reported runs use, so a parameter picked here means the same thing there:

    source ~/efd/.venv/bin/activate
    python src/scripts/tune_persistence.py --predictions ~/predictions/tslm_A-10.jsonl \
        --care-root ~/data/CARE_To_Compare

NO-LEAKAGE RULE (PLAN.md 5.1): run this on the fold's *validation* turbine only -- A-0's fold holds
out A-10 for early stopping, so the model never took a gradient step on it. Never run it on the
test turbine: the parameters would then be fitted to the number we report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.care_score import care_root, evaluate, jsonable, load_predictions  # noqa: E402

from energy_fault_detector.evaluation import Care2CompareDataset  # noqa: E402

DEFAULT_WINDOWS = (1, 6, 12, 18, 24, 36, 48, 72)
DETECTION_CRITICALITY = 72
"""CARE counts an anomaly event as detected above this; 72 10-minute stamps is 12 hours."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="write the whole sweep as JSON")
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.17, 0.25, 0.34, 0.5],
        help="k as a fraction of N (rounded up, at least 1); k=1 is always included, duplicates dropped",
    )
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    turbines = {r.get("turbine") for rows in predictions.values() for r in rows}
    print(f"{len(predictions)} events of {sorted(turbines)}\n")

    dataset = Care2CompareDataset(path=care_root(args.care_root))
    cache: dict = {}
    grid = []
    for window in sorted(set(args.windows)):
        ks = {1} | {max(1, min(window, -(-int(round(f * window * 100)) // 100))) for f in args.fractions}
        for k in sorted(ks):
            grid.append((window, k))

    print(f"{'N':>4} {'k':>3} {'CARE':>7} {'cov':>6} {'acc':>6} {'rel':>6} {'early':>6}   per-event max criticality")
    sweep = []
    for window, k in grid:
        summary, per_event = evaluate(
            predictions,
            dataset,
            persist_window_hours=window,
            persist_needed_hours=k,
            cache=cache,
            verbose=False,
        )
        crit = " ".join(
            f"{e['farm']}{e['event_id']}{'*' if e['event_label'] == 'anomaly' else ''}"
            f"={e['max_criticality']:.0f}"
            for e in per_event
        )
        detected = sum(
            e["event_label"] == "anomaly" and e["max_criticality"] > DETECTION_CRITICALITY for e in per_event
        )
        false_detected = sum(
            e["event_label"] == "normal" and e["max_criticality"] > DETECTION_CRITICALITY for e in per_event
        )
        sweep.append(summary | {"detected_anomalies": detected, "detected_normals": false_detected, "events": per_event})
        print(
            f"{window:>4} {k:>3} {summary['care_score']:>7.4f} {summary['coverage']:>6.3f} "
            f"{summary['accuracy']:>6.3f} {summary['reliability']:>6.3f} {summary['earliness']:>6.3f}   {crit}"
        )

    best = max(sweep, key=lambda s: s["care_score"])
    print(
        f"\nbest on this turbine: N={best['persist_window_hours']} k={best['persist_needed_hours']} "
        f"-> CARE {best['care_score']:.4f} "
        f"({best['detected_anomalies']} anomaly events detected, {best['detected_normals']} normal events wrongly)"
    )
    print("* marks anomaly events. Detection needs max criticality > 72 (about 12 hours sustained).")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sweep, indent=2, default=jsonable))
        print(f"sweep written to {args.out}")


if __name__ == "__main__":
    main()
