"""Summarize the multi-fold evaluation exactly as pre-registered in PLAN.md 8.12.

Runs in the energy-fault-detector environment, on the per-fold prediction files of
``src/scripts/run_folds.sh`` and of the residual baseline:

    source ~/efd/.venv/bin/activate
    python src/scripts/summarize_folds.py --tslm '~/predictions/folds/tslm_*_last.jsonl' \
        --baseline '~/predictions/folds/baseline_*.jsonl' --care-root ~/data/CARE_To_Compare \
        --out ~/results/folds/summary.json

Every number uses the same ``evaluate`` and the same alarm persistence (24, 1) for both methods. The
main result is the pooled score over the clean folds; the development folds (A-0 picked the answer
format, A-10 the persistence) are reported on their own line and never pooled into it. CARE-Score
alone can be gamed by hardly ever alarming, so the detected anomaly events and the falsely detected
normal events are reported next to it.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.care_score import care_root, evaluate, jsonable, load_predictions  # noqa: E402

from energy_fault_detector.evaluation import Care2CompareDataset  # noqa: E402

DEV_FOLDS = ("A-0", "A-10")
DETECTION_CRITICALITY = 72
"""CARE counts an event as detected above this (about 12 hours of sustained alarms)."""


def by_turbine(pattern: str) -> dict[str, dict]:
    """Predictions grouped by event, per test turbine."""
    files = sorted(glob.glob(str(Path(pattern).expanduser())))
    out: dict[str, dict] = {}
    for path in files:
        for key, rows in load_predictions([path]).items():
            out.setdefault(rows[0]["turbine"], {})[key] = rows
    return out


def pooled(per_turbine: dict[str, dict], turbines: list[str]) -> dict:
    return {key: rows for t in turbines for key, rows in per_turbine.get(t, {}).items()}


def counts(per_event: list[dict]) -> dict:
    anomalies = [e for e in per_event if e["event_label"] == "anomaly"]
    normals = [e for e in per_event if e["event_label"] == "normal"]
    return {
        "anomaly_events": len(anomalies),
        "detected": sum(e["max_criticality"] > DETECTION_CRITICALITY for e in anomalies),
        "normal_events": len(normals),
        "false_detected": sum(e["max_criticality"] > DETECTION_CRITICALITY for e in normals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tslm", required=True, help="glob of the per-fold TSLM prediction files")
    parser.add_argument("--baseline", required=True, help="glob of the per-fold baseline prediction files")
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = Care2CompareDataset(path=care_root(args.care_root))
    methods = {"TSLM last": by_turbine(args.tslm), "baseline": by_turbine(args.baseline)}
    done = sorted(set(methods["TSLM last"]) & set(methods["baseline"]))
    missing = sorted(set(methods["TSLM last"]) ^ set(methods["baseline"]))
    clean = [t for t in done if t not in DEV_FOLDS]
    groups = {
        f"clean, all ({len(clean)} folds)": clean,
        "clean, farm A": [t for t in clean if t.startswith("A")],
        "clean, farm B": [t for t in clean if t.startswith("B")],
        **{f"dev {t}": [t] for t in DEV_FOLDS if t in done},
    }
    print(f"folds with both methods: {done}")
    if missing:
        print(f"folds with only one method (left out): {missing}")

    cache: dict = {}
    report: dict = {"groups": {}, "events": {}}
    print(f"\n{'group':<24} {'method':<10} {'CARE':>7} {'cov':>6} {'acc':>6} {'rel':>6} {'early':>6} "
          f"{'detected':>9} {'false':>7}")
    for name, turbines in groups.items():
        if not turbines:
            continue
        for method, per_turbine in methods.items():
            summary, per_event = evaluate(
                pooled(per_turbine, turbines), dataset,
                persist_window_hours=24, persist_needed_hours=1, cache=cache, verbose=False,
            )
            c = counts(per_event)
            report["groups"].setdefault(name, {})[method] = summary | c
            if name.startswith("clean, all"):
                report["events"][method] = per_event
            print(
                f"{name:<24} {method:<10} {summary['care_score']:>7.4f} {summary['coverage']:>6.3f} "
                f"{summary['accuracy']:>6.3f} {summary['reliability']:>6.3f} {summary['earliness']:>6.3f} "
                f"{c['detected']:>4}/{c['anomaly_events']:<4} {c['false_detected']:>3}/{c['normal_events']:<3}"
            )
        print()

    if report["events"]:
        print("clean folds, per event (crit > 72 = detected; lead = days before the event start)")
        rows = {m: {(e["farm"], int(e["event_id"])): e for e in evs} for m, evs in report["events"].items()}
        print(f"{'event':<7} {'turbine':<7} {'label':<8} {'TSLM crit':>9} {'lead':>6} {'base crit':>9} {'lead':>6}")
        for key in sorted(rows["TSLM last"]):
            t, b = rows["TSLM last"][key], rows["baseline"].get(key, {})

            def fmt(e: dict) -> tuple[str, str]:
                if not e:
                    return "-", "-"
                lead = e.get("first_alarm_lead_days")
                return f"{e['max_criticality']:.0f}", ("-" if lead is None else f"{lead:.1f}")

            (tc, tl), (bc, bl) = fmt(t), fmt(b)
            print(f"{key[0]}{key[1]:<6} {t.get('turbine', ''):<7} {t['event_label']:<8} {tc:>9} {tl:>6} {bc:>9} {bl:>6}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=jsonable))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
