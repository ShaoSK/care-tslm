"""Compare any number of prediction sets on the same folds, with the CARE scoring of PLAN.md 8.12.

Runs in the energy-fault-detector environment:

    python src/scripts/compare_methods.py --care-root ~/data/CARE_To_Compare \
        --folds B-13 B-11 B-12 A-13 \
        --method "last=~/predictions/folds/tslm_{T}_last.jsonl" \
        --method "baseline=~/predictions/folds/baseline_{T}.jsonl" \
        --method "v4 cal0.95=~/predictions/v4/cal0.95_{T}.jsonl" \
        --out ~/results/v4/compare.json

``{T}`` in a path is replaced by each fold's turbine. Only folds that every method has are pooled
(the rest are listed). Same ``evaluate``, persistence (24, 1) and detection rule (crit > 72) as
``summarize_folds.py``, so the numbers line up with the pre-registered summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.care_score import care_root, evaluate, jsonable, load_predictions  # noqa: E402
from scripts.summarize_folds import counts  # noqa: E402

from energy_fault_detector.evaluation import Care2CompareDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--method", action="append", required=True, help="'name=path with {T}'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    methods = dict(m.split("=", 1) for m in args.method)
    paths = {
        name: {t: Path(pattern.replace("{T}", t)).expanduser() for t in args.folds} for name, pattern in methods.items()
    }
    folds = [t for t in args.folds if all(paths[m][t].exists() for m in methods)]
    left_out = [t for t in args.folds if t not in folds]
    print(f"folds compared: {folds}" + (f"; left out (a method has no file): {left_out}" if left_out else ""))
    if not folds:
        raise SystemExit("no fold has every method")

    dataset = Care2CompareDataset(path=care_root(args.care_root))
    cache: dict = {}
    report: dict = {"folds": folds, "methods": {}, "events": {}}
    width = max(len(m) for m in methods)
    print(f"\n{'method':<{width}} {'CARE':>7} {'cov':>6} {'acc':>6} {'rel':>6} {'early':>6} {'detected':>9} {'false':>7}")
    for name in methods:
        preds = load_predictions([str(paths[name][t]) for t in folds])
        summary, per_event = evaluate(
            preds, dataset, persist_window_hours=24, persist_needed_hours=1, cache=cache, verbose=False
        )
        c = counts(per_event)
        report["methods"][name] = summary | c
        report["events"][name] = per_event
        print(
            f"{name:<{width}} {summary['care_score']:>7.4f} {summary['coverage']:>6.3f} {summary['accuracy']:>6.3f} "
            f"{summary['reliability']:>6.3f} {summary['earliness']:>6.3f} "
            f"{c['detected']:>4}/{c['anomaly_events']:<4} {c['false_detected']:>3}/{c['normal_events']:<3}"
        )

    print("\nper event: max criticality (crit > 72 = detected)")
    by_key = {m: {(e["farm"], int(e["event_id"])): e for e in evs} for m, evs in report["events"].items()}
    first = next(iter(by_key.values()))
    print(f"{'event':<7} {'turbine':<8} {'label':<8} " + " ".join(f"{m[:12]:>12}" for m in methods))
    for key in sorted(first):
        e = first[key]
        cells = " ".join(f"{by_key[m].get(key, {}).get('max_criticality', float('nan')):>12.0f}" for m in methods)
        print(f"{key[0]}{key[1]:<6} {e.get('turbine', ''):<8} {e['event_label']:<8} {cells}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=jsonable))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
