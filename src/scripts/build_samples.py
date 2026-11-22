"""Build the snapshot store.

    # From the extracted release (development path):
    python src/scripts/build_samples.py --raw /data/CARE_To_Compare --farms A B --out data/store_ab

    # From the TimeNet dataset built by our connector:
    python src/scripts/build_samples.py --timenet /data/timenet/care_to_compare --farms A B --out data/store_ab
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.build import build_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="extracted CARE_To_Compare directory")
    source.add_argument("--timenet", type=Path, help="local TimeNet dataset version directory")
    parser.add_argument("--farms", nargs="+", default=["A", "B"])
    parser.add_argument("--events", nargs="*", type=int, help="only these event IDs (for quick runs)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--with-raw",
        action="store_true",
        help="carry every component channel a second time as its raw reading (PLAN.md 5.4): "
        "doubles the component curves, needs its own store directory",
    )
    parser.add_argument(
        "--with-pairs",
        action="store_true",
        help="carry every same-group sensor pair's residual difference as an extra channel "
        "(config.sensor_pairs; A 2 pairs, B 6): needs its own store directory",
    )
    args = parser.parse_args()

    if args.raw:
        from care_tslm.load_raw import load_farm

        events = (e for farm in args.farms for e in load_farm(args.raw, farm, args.events))
    else:
        from care_tslm.load_timenet import load_dataset

        events = (e for e in load_dataset(args.timenet, tuple(args.farms)) if not args.events or e.event_id in args.events)
    build_store(events, args.out, with_raw=args.with_raw, with_pairs=args.with_pairs)


if __name__ == "__main__":
    main()
