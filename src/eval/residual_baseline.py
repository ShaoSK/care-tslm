"""Residual-threshold baseline (PLAN.md 8.3): alarm when any component group deviates.

No model and no GPU: it applies exactly the rule that writes the training rationales -- a group
counts as deviating when its largest |residual| exceeds a high quantile of the *training* fold's
normal snapshots (``text.fit_thresholds``) -- and calls that an anomaly. So it isolates how much
the fine-tuned TSLM adds on top of the features it is given.

Output is the same JSON-line format as ``src/train/predict.py``, so the same scorer runs on both:

    python src/eval/residual_baseline.py --store ~/data/store_ab --test-turbines A-0 \
        --out predictions/baseline_A-0.jsonl
    python src/eval/care_score.py --predictions predictions/baseline_A-0.jsonl \
        --care-root ~/data/CARE_To_Compare --out results/care_score_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import folds, text  # noqa: E402
from care_tslm.build import load_meta  # noqa: E402
from care_tslm.config import STEPS_PER_HOUR  # noqa: E402

KEYS = ("farm", "event_id", "turbine", "end_step", "end_us", "status_end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--test-turbines", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-stride-hours", type=int, default=1)
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="multiplies every threshold: >1 is a less trigger-happy alarm, <1 a more sensitive one",
    )
    args = parser.parse_args()

    meta = load_meta(args.store)
    test_set = set(args.test_turbines)
    unknown = test_set - set(folds.turbines(meta))
    if unknown:
        raise SystemExit(f"test turbines {sorted(unknown)} are not in the store")

    # Thresholds from the training-side turbines only -- the same no-leakage rule as the rationales.
    thresholds = text.fit_thresholds(m for m in meta if m["turbine"] not in test_set)
    thresholds = {k: v * args.threshold_scale for k, v in thresholds.items()}

    stride = args.test_stride_hours * STEPS_PER_HOUR
    rows = [m for m in folds.test_rows(meta, test_set) if m["end_step"] % stride == 0]
    rows.sort(key=lambda m: (m["farm"], m["event_id"], m["end_step"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"anomaly": 0, "normal": 0}
    with args.out.open("w") as out:
        for m in rows:
            found = text.deviations(m, thresholds)
            pred = "anomaly" if found else "normal"
            counts[pred] += 1
            row = {k: m[k] for k in KEYS} | {
                "label": m["target"],
                "pred": pred,
                "generated": f"groups={[d['group'] for d in found]} Answer: {pred}",
            }
            out.write(json.dumps(row) + "\n")

    gold_anomaly = sum(m["target"] == "anomaly" for m in rows)
    hit = sum(1 for m in rows if m["target"] == "anomaly" and text.deviations(m, thresholds))
    print(
        f"{len(rows)} test snapshots -> {args.out}\n"
        f"  alarms: {counts['anomaly']} ({counts['anomaly'] / max(1, len(rows)):.1%})\n"
        f"  gold anomaly snapshots: {gold_anomaly}, of which alarmed: {hit} "
        f"({hit / max(1, gold_anomaly):.1%} snapshot-level recall)"
    )


if __name__ == "__main__":
    main()
