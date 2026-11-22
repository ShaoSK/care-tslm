"""Build the snapshot store for a set of events: one .npz per event plus a metadata index.

Layout of ``out_dir``:

- ``events.json``: per event, label, windows, channel texts, residual-model fit sizes;
- ``samples.jsonl``: one line per snapshot (see :func:`care_tslm.snapshots.build`);
- ``arrays/<farm><event>.npz``: ``long`` (n, C, 180) and ``short`` (n, C, 168), float16.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import time

import numpy as np

from care_tslm import residuals, snapshots
from care_tslm.events import EventData


def build_store(
    events: Iterable[EventData], out_dir: Path, *, with_raw: bool = False, with_pairs: bool = False, log=print
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(exist_ok=True)
    index = {}
    with (out_dir / "samples.jsonl").open("w") as samples:
        for event in events:
            started = time.time()
            res = residuals.compute(event)
            snaps = snapshots.build(event, res, with_raw=with_raw, with_pairs=with_pairs)
            np.savez_compressed(out_dir / "arrays" / f"{event.key}.npz", long=snaps.long, short=snaps.short)
            for meta in snaps.meta:
                samples.write(json.dumps(meta) + "\n")
            index[event.key] = {
                "farm": event.farm,
                "event_id": event.event_id,
                "asset_id": event.asset_id,
                "turbine": event.turbine,
                "label": event.label,
                "description": event.description,
                "fault_component": event.fault_component,
                "n_steps": event.n_steps,
                "train_steps": event.train_steps,
                "baseline_end": event.baseline_end,
                "prediction_steps": event.prediction_steps,
                "window_steps": event.window_steps,
                "channels": snaps.channels,
                "roles": snaps.roles,
                "channel_text": event.channel_text,
                "fit_points": res.fit_points,
                "baseline": res.baseline,
                "pair_baseline": res.pair_baseline,
                "load_baseline": res.load_baseline,
                "n_snapshots": len(snaps.meta),
            }
            pools = {p: sum(m["train_pool"] == p for m in snaps.meta) for p in ("anomaly", "normal", None)}
            log(
                f"{event.key:>5} {event.label:<7} steps={event.n_steps:>6} snapshots={len(snaps.meta):>5} "
                f"pool anomaly={pools['anomaly']} normal={pools['normal']} unused={pools[None]} "
                f"no-model={[s for s, n in res.fit_points.items() if n == 0]} ({time.time() - started:.1f}s)"
            )
    (out_dir / "events.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


def load_meta(store: Path) -> list[dict]:
    with (store / "samples.jsonl").open() as f:
        return [json.loads(line) for line in f]


def load_events(store: Path) -> dict[str, dict]:
    return json.loads((store / "events.json").read_text())


class ArrayCache:
    """Lazy per-event access to the snapshot arrays."""

    def __init__(self, store: Path) -> None:
        self.store = store
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def get(self, meta: dict) -> tuple[np.ndarray, np.ndarray]:
        key = f"{meta['farm']}{meta['event_id']}"
        if key not in self._cache:
            with np.load(self.store / "arrays" / f"{key}.npz") as data:
                self._cache[key] = {"long": data["long"], "short": data["short"]}
        arrays = self._cache[key]
        return arrays["long"][meta["index"]], arrays["short"][meta["index"]]
