"""Print what a snapshot store contains: counts, one fold's training set, example texts, signal check.

    python src/scripts/inspect_samples.py data/store_ab [--test-turbine A-0] [--examples 3]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import folds, text  # noqa: E402
from care_tslm.build import ArrayCache, load_events, load_meta  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("--test-turbine", help="the held-out turbine of the fold to show (default: first)")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    meta = load_meta(args.store)
    events = load_events(args.store)
    print(f"{len(events)} events, {len(meta)} snapshots")
    print("by farm/role/pool:", dict(Counter((m["farm"], m["role"], m["train_pool"]) for m in meta)))

    all_turbines = folds.turbines(meta)
    test = {args.test_turbine or all_turbines[0]}
    train = folds.train_rows(meta, test)
    test_set = folds.test_rows(meta, test)
    thresholds = text.fit_thresholds(m for m in meta if m["turbine"] not in test)
    print(f"\nfold: test turbine {sorted(test)} | train {len(train)} "
          f"({Counter(m['target'] for m in train)}) | test {len(test_set)} ({Counter(m['target'] for m in test_set)})")
    print("thresholds (|z|):", {f"{f}/{g}/{s}": round(v, 2) for (f, g, s), v in sorted(thresholds.items())})

    # How often does the rule-based rationale agree with the label?
    consistency = Counter()
    for m in train:
        _, facts = text.rationale(m, events[f"{m['farm']}{m['event_id']}"]["channel_text"], thresholds)
        consistency[(m["target"], facts["consistent"])] += 1
    print("\nrationale vs label on the training set (target, has-matching-evidence):", dict(consistency))

    # Per anomaly event: strongest group in the window, compared with the normal pool.
    print("\nper anomaly event, max group |z| (30d mean) inside the window:")
    for key, info in events.items():
        if info["label"] != "anomaly":
            continue
        rows = [m for m in meta if f"{m['farm']}{m['event_id']}" == key and m["in_window"]]
        if not rows:
            print(f"  {key}: no snapshot in window")
            continue
        best = Counter()
        for m in rows:
            for group, per_stat in text.group_scores(m).items():
                if "mean_30d" in per_stat:
                    best[group] = max(best[group], abs(per_stat["mean_30d"][1]))
        top = ", ".join(f"{g} {v:.1f}" for g, v in best.most_common(3))
        print(f"  {key} ({info['description']}): {top}")

    cache = ArrayCache(args.store)
    rng = random.Random(1)
    picks = [m for m in train if m["target"] == "anomaly"][:1] + rng.sample(train, min(args.examples, len(train)))
    for m in picks[: args.examples]:
        info = events[f"{m['farm']}{m['event_id']}"]
        long, short = cache.get(m)
        prompts = text.channel_prompts(m["farm"], long, short, info["channel_text"])
        answer, _ = text.rationale(m, info["channel_text"], thresholds)
        print("\n" + "=" * 100)
        print(f"{m['farm']}{m['event_id']} end_step={m['end_step']} target={m['target']} role={m['role']}")
        print(f"[{len(prompts)} series, lengths {sorted({len(s) for _, s in prompts})}]")
        for t, s in prompts[:2] + prompts[3:5]:
            print(f"  - {t}  [{s[:4].round(2)} ...]")
        print("  ...")
        print(f"Rationale: {answer}")


if __name__ == "__main__":
    main()
