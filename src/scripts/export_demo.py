"""Export what the demo page (docs/) shows, per event, into one JSON file.

Runs in the energy-fault-detector environment (it uses CARE's own criticality):

    python src/scripts/export_demo.py --store ~/data/store_ab_v4 --care-root ~/data/CARE_To_Compare \
        --folds B-13 B-11 B-12 A-13 \
        --method "baseline=~/predictions/folds/baseline_{T}.jsonl" \
        --method "OpenTSLM (pre-registered)=~/predictions/folds/tslm_{T}_last.jsonl" \
        --method "OpenTSLM v4=~/predictions/v4/cal0.95_{T}.jsonl" \
        --explanations ~/predictions/v4/explain.jsonl --out docs/data/demo.json

Per event: its window, hourly residual curves of every component sensor and sensor pair (the last
hour of each hourly snapshot, i.e. what the model's 7.5-day curve ends on), and per method the hourly
P(anomaly) if there is one, the alarm after persistence (24, 1) and CARE's criticality, computed
exactly as ``eval/care_score.py`` scores it. Nothing here is used for any result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.build import ArrayCache, load_events, load_meta  # noqa: E402
from care_tslm.config import PAIR_ROLE  # noqa: E402
from care_tslm.scoring import alarm_series, persist  # noqa: E402
from eval.care_score import care_root, event_bounds, load_event  # noqa: E402

from energy_fault_detector.evaluation import Care2CompareDataset  # noqa: E402
from energy_fault_detector.utils.analysis import calculate_criticality  # noqa: E402

PERSIST = (24, 1)
IGNORE_NORMAL_INDEX_FARMS = ("A",)


def rounded(values, digits=2):
    return [None if v is None or not np.isfinite(v) else round(float(v), digits) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--care-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--method", action="append", required=True, help="'name=path with {T}'")
    parser.add_argument("--explanations", type=Path, default=None, help="predict.py output with 'generated'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = load_meta(args.store)
    events = load_events(args.store)
    cache = ArrayCache(args.store)
    dataset = Care2CompareDataset(path=care_root(args.care_root))
    methods = dict(m.split("=", 1) for m in args.method)

    explanations: dict[tuple[str, int], list[dict]] = defaultdict(list)
    if args.explanations and args.explanations.exists():
        for row in map(json.loads, args.explanations.open()):
            explanations[(row["farm"], int(row["event_id"]))].append(
                {"t": int(row["end_us"] // 1_000_000), "label": row["label"], "pred": row["pred"], "text": row["generated"]}
            )

    out = {"persistence_hours": list(PERSIST), "detection_criticality": 72, "methods": list(methods), "events": []}
    csv_cache: dict = {}
    for turbine in args.folds:
        rows_by_method = {}
        for name, pattern in methods.items():
            path = Path(pattern.replace("{T}", turbine)).expanduser()
            by_event = defaultdict(list)
            for row in map(json.loads, path.open()):
                by_event[(row["farm"], int(row["event_id"]))].append(row)
            rows_by_method[name] = by_event
        keys = sorted({k for by_event in rows_by_method.values() for k in by_event})
        for farm, event_id in keys:
            info = events[f"{farm}{event_id}"]
            snaps = sorted(
                (m for m in meta if m["farm"] == farm and m["event_id"] == event_id and m["role"] == "prediction"),
                key=lambda m: m["end_step"],
            )
            times = [int(m["end_us"] // 1_000_000) for m in snaps]
            curves = []
            last_points = np.stack([cache.get(m)[1][:, -1].astype(np.float64) for m in snaps])  # (hours, channels)
            for i, (name, role) in enumerate(zip(info["channels"], info["roles"], strict=True)):
                if role not in ("component", PAIR_ROLE):
                    continue
                if role == PAIR_ROLE:
                    a, b = name.split("-")
                    label = f"{info['channel_text'][a]} minus {info['channel_text'][b]}"
                else:
                    label = info["channel_text"][name]
                curves.append({"name": name, "role": role, "label": label, "values": rounded(last_points[:, i])})

            care_info = dataset.get_event_info(event_id)
            test = load_event(dataset, event_id, csv_cache)
            start, end = event_bounds(care_info, test.index)
            normal_index = test["status_type_id"] == 0
            if farm in IGNORE_NORMAL_INDEX_FARMS:
                normal_index = pd.Series(True, index=test.index)
            hourly_index = pd.to_datetime([t * 1_000_000 for t in times], unit="us")

            per_method = {}
            for name, by_event in rows_by_method.items():
                rows = by_event.get((farm, event_id))
                if not rows:
                    continue
                judged = persist(rows, window_hours=PERSIST[0], needed_hours=PERSIST[1])
                alarms, _ = alarm_series(judged, test.index)
                crit = calculate_criticality(alarms, normal_index)
                crit_hourly = crit.reindex(hourly_index, method="ffill").fillna(0)
                by_time = {int(r["end_us"] // 1_000_000): r for r in judged}
                per_method[name] = {
                    "p_anomaly": rounded([by_time.get(t, {}).get("p_anomaly") for t in times], 4)
                    if any("p_anomaly" in r for r in rows)
                    else None,
                    "threshold": rows[0].get("threshold"),
                    "raw_alarm": [int(by_time.get(t, {}).get("pred_raw") == "anomaly") for t in times],
                    "alarm": [int(by_time.get(t, {}).get("pred") == "anomaly") for t in times],
                    "criticality": [int(v) for v in crit_hourly.to_numpy()],
                    "max_criticality": int(crit.loc[: pd.Timestamp(end)].max()) if len(crit) else 0,
                }
            out["events"].append(
                {
                    "key": f"{farm}{event_id}",
                    "farm": farm,
                    "turbine": turbine,
                    "label": info["label"],
                    "fault_component": info.get("fault_component"),
                    "description": info.get("description"),
                    "window": [int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())],
                    "times": times,
                    "curves": curves,
                    "methods": per_method,
                    "explanations": sorted(explanations.get((farm, event_id), []), key=lambda e: e["t"]),
                }
            )
            print(f"{farm}{event_id}: {len(times)} hours, {len(curves)} curves, methods {list(per_method)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"written to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
