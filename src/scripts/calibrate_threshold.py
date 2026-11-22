"""Turn P(anomaly) scores into alarms with a threshold from the test turbine's OWN healthy period.

PLAN.md 8.13, option 3. ``score_answers.py --calibration`` scores the test turbine's training-period
snapshots (the second half of each training year: healthy by CARE's definition, never trained on;
CARE gives exactly this period to every method to learn the turbine's normal behaviour). The threshold
is the ``--quantile`` of those scores, so the turbine alarms on about (1 - q) of its healthy hours
whatever its overall offset -- a turbine that 'runs warm' to the model is calibrated away.

No labels are read: neither the calibration rows' (all normal) nor the test rows'. Pure numpy, so it
runs in either environment:

    python src/scripts/calibrate_threshold.py --calib ~/predictions/v4/calib_B-13.jsonl \
        --test ~/predictions/v4/scored_B-13.jsonl --out-dir ~/predictions/v4

Writes ``<out-dir>/cal<q>_<turbine>.jsonl`` for each quantile (rows with ``pred`` from the threshold,
the argmax kept as ``pred_greedy``) and prints each one's threshold and alarm rate. The pre-registered
main quantile is 0.95; the others are a sensitivity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.scoring import calibrated_threshold, rank_auc, threshold_rows  # noqa: E402

MAIN_QUANTILE = 0.95


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", type=Path, required=True, help="score_answers.py --calibration output")
    parser.add_argument("--test", type=Path, required=True, help="score_answers.py output on the test split")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.9, MAIN_QUANTILE, 0.99])
    args = parser.parse_args()

    calib, test = read(args.calib), read(args.test)
    turbines = {r["turbine"] for r in test}
    if len(turbines) != 1 or {r["turbine"] for r in calib} != turbines:
        raise SystemExit(f"expected one and the same turbine in both files, got {turbines} / {sorted({r['turbine'] for r in calib})}")
    turbine = turbines.pop()
    p_calib = [r["p_anomaly"] for r in calib]
    auc = rank_auc([r["p_anomaly"] for r in test], [r["label"] == "anomaly" for r in test])
    greedy = sum(r["pred"] == "anomaly" for r in test) / len(test)
    print(f"{turbine}: {len(calib)} calibration rows, {len(test)} test rows; greedy alarm rate {greedy:.3f}; "
          f"snapshot AUC {auc if auc is None else round(auc, 3)} (diagnostic only)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for q in args.quantiles:
        threshold = calibrated_threshold(p_calib, q)
        rows = threshold_rows(test, threshold)
        rate = sum(r["pred"] == "anomaly" for r in rows) / len(rows)
        out = args.out_dir / f"cal{q:g}_{turbine}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row | {"threshold": threshold}) + "\n")
        print(f"  q={q:g}{' (main)' if q == MAIN_QUANTILE else ''}: threshold {threshold:.4g}, "
              f"test alarm rate {rate:.3f} -> {out}")


if __name__ == "__main__":
    main()
