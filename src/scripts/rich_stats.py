"""A rich statistic pack per snapshot, for a teacher model to read (PLAN.md 6, teacher labelling).

Beyond the three residual means the rationale rules use, this adds slope, volatility, extremes and
acceleration, and extends the same-condition residual to every numeric column of the release --
including the per-10-minute std/min/max that only nine of wind farm A's sensors carry, and the
sensors the channel table leaves out. Every figure is in sigma of that column's baseline residual,
so they are comparable across sensors and farms.

    python src/scripts/rich_stats.py --raw ~/data/CARE_To_Compare --store ~/data/store_ab \
        --out ~/results/rich_stats.json

Selection: a few end times per event (``--per-event``), spread over the event window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.config import (  # noqa: E402
    CHANNELS,
    MIN_BASELINE_POINTS,
    OPERATING_POWER,
    STEPS_PER_DAY,
    STEPS_PER_HOUR,
    Z_CLIP,
)
from care_tslm.load_raw import _source_module, farm_directory  # noqa: E402

SKIP_PREFIX = ("time_stamp", "asset_id", "id", "train_test", "status_type_id")


def describe(source, sensors: dict, column: str) -> dict:
    """Sensor description, unit and statistic of one column.

    A few columns of the release carry no statistic suffix (``sensor_44``); they are kept as-is.
    """
    try:
        name, statistic = source.split_feature_column(column)
    except Exception:
        name, statistic = column, "unknown"
    info = sensors.get(name)
    clean = lambda text: text.replace("\ufffd", "°")  # noqa: E731
    return {
        "description": clean(info.description) if info else column,
        "unit": clean(info.unit) if info else "",
        "statistic": statistic,
    }


def role_column(farm: str, role: str) -> str:
    return next(f"{c.sensor}_avg" for c in CHANNELS[farm] if c.role == role)


def residual_z(y: np.ndarray, design: np.ndarray, fit_rows: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """The same-condition residual of one column, in sigma (``residuals.compute``, generalized)."""
    rows = fit_rows & np.isfinite(y)
    if int(rows.sum()) < MIN_BASELINE_POINTS:
        return None
    mu, sd = design[rows].mean(axis=0), design[rows].std(axis=0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd == 0] = 1.0
    x = (design - mu) / sd
    weights, *_ = np.linalg.lstsq(x[rows], y[rows], rcond=None)
    residual = y - x @ weights
    scale = float(np.std(residual[rows]))
    if scale <= 0:
        return None
    out = np.full(y.shape, np.nan, dtype=np.float64)
    keep = valid & np.isfinite(y)
    out[keep] = np.clip(residual[keep] / scale, -Z_CLIP, Z_CLIP)
    return out


def window_stats(z: np.ndarray, end: int) -> dict:
    """Level, spread, extreme and trend of one residual series over three look-backs."""
    out: dict[str, float | None] = {}
    for label, days in (("30d", 30), ("7d", 7), ("24h", 1)):
        lo = max(0, end + 1 - days * STEPS_PER_DAY)
        piece = z[lo : end + 1]
        finite = piece[np.isfinite(piece)]
        if finite.size < 6:
            out |= {f"mean_{label}": None, f"std_{label}": None, f"absmax_{label}": None}
            continue
        out[f"mean_{label}"] = round(float(finite.mean()), 2)
        out[f"std_{label}"] = round(float(finite.std()), 2)
        out[f"absmax_{label}"] = round(float(np.abs(finite).max()), 2)
    for label, days in (("30d", 30), ("7d", 7)):
        lo = max(0, end + 1 - days * STEPS_PER_DAY)
        piece = z[lo : end + 1]
        idx = np.flatnonzero(np.isfinite(piece))
        if idx.size < 20:
            out[f"slope_{label}"] = None
            continue
        t = idx / STEPS_PER_DAY
        out[f"slope_{label}"] = round(float(np.polyfit(t, piece[idx], 1)[0]), 3)
    for a, b in (("24h", "7d"), ("7d", "30d")):
        va, vb = out.get(f"mean_{a}"), out.get(f"mean_{b}")
        out[f"delta_{a}_vs_{b}"] = None if va is None or vb is None else round(va - vb, 2)
    return out


def raw_window_stats(y: np.ndarray, valid: np.ndarray, end: int) -> dict:
    """The column's own readings in physical units over three look-backs, operating steps only."""
    out: dict[str, float | None] = {}
    for label, days in (("30d", 30), ("7d", 7), ("24h", 1)):
        lo = max(0, end + 1 - days * STEPS_PER_DAY)
        piece = y[lo : end + 1][valid[lo : end + 1]]
        piece = piece[np.isfinite(piece)]
        if piece.size < 6:
            out |= {f"raw_mean_{label}": None, f"raw_std_{label}": None, f"raw_p05_{label}": None, f"raw_p95_{label}": None}
            continue
        out[f"raw_mean_{label}"] = round(float(piece.mean()), 2)
        out[f"raw_std_{label}"] = round(float(piece.std()), 2)
        out[f"raw_p05_{label}"] = round(float(np.percentile(piece, 5)), 2)
        out[f"raw_p95_{label}"] = round(float(np.percentile(piece, 95)), 2)
    return out


def baseline_reference(y: np.ndarray, fit_rows: np.ndarray) -> dict:
    """What this column reads on this turbine's own HEALTHY baseline: the 'normal standard'."""
    piece = y[fit_rows]
    piece = piece[np.isfinite(piece)]
    if piece.size < MIN_BASELINE_POINTS:
        return {"baseline_n": int(piece.size)}
    q = np.percentile(piece, [5, 25, 50, 75, 95])
    return {
        "baseline_n": int(piece.size),
        "baseline_mean": round(float(piece.mean()), 2),
        "baseline_std": round(float(piece.std()), 2),
        "baseline_p05": round(float(q[0]), 2),
        "baseline_p25": round(float(q[1]), 2),
        "baseline_median": round(float(q[2]), 2),
        "baseline_p75": round(float(q[3]), 2),
        "baseline_p95": round(float(q[4]), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="for the event windows and labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--farms", nargs="+", default=["A", "B"])
    parser.add_argument("--per-event", type=int, default=4, help="snapshot end times per event")
    args = parser.parse_args()

    source = _source_module()
    from care_tslm.build import load_events

    store_events = load_events(args.store)
    packs = []
    for farm in args.farms:
        directory = farm_directory(args.raw, farm)
        sensors = source.read_sensor_descriptions(directory / "feature_description.csv")
        infos = source.read_event_info(directory / "event_info.csv")
        for event_id in sorted(infos):
            key = f"{farm}{event_id}"
            if key not in store_events:
                continue
            info = store_events[key]
            table = source.read_event_table(directory / "datasets" / f"{event_id}.csv")
            index = {c: i for i, c in enumerate(table.feature_columns)}
            n = table.values.shape[1]

            ambient = table.values[index[role_column(farm, "ambient")]].astype(np.float64)
            power = table.values[index[role_column(farm, "power")]].astype(np.float64)
            wind = table.values[index[role_column(farm, "wind")]].astype(np.float64)
            t0, _ = info["train_steps"]
            base = np.zeros(n, dtype=bool)
            base[t0 : info["baseline_end"]] = True
            base &= table.status == 0
            rated = np.nanpercentile(power[base], 99) if np.isfinite(power[base]).any() else np.nan
            if not np.isfinite(rated) or rated <= 0:
                rated = float(np.nanmax(np.abs(power))) or 1.0
            fraction = power / rated
            operating = np.nan_to_num(fraction, nan=0.0) > OPERATING_POWER
            design = np.stack([np.ones(n), ambient, fraction, fraction**2, wind], axis=1)
            usable = np.isfinite(design).all(axis=1)
            fit_rows = base & operating & usable
            valid = operating & usable

            zs: dict[str, np.ndarray] = {}
            raws: dict[str, np.ndarray] = {}
            for column, i in index.items():
                if column.startswith(SKIP_PREFIX):
                    continue
                y = table.values[i].astype(np.float64)
                z = residual_z(y, design, fit_rows, valid)
                if z is not None:
                    zs[column] = z
                    raws[column] = y

            p0, p1 = info["prediction_steps"]
            w0, w1 = info["window_steps"]
            ends = np.linspace(max(p0, w0), min(p1, w1), args.per_event, dtype=int)
            in_model = {f"{c.sensor}_avg" for c in CHANNELS[farm]}
            for end in sorted({int(e) for e in ends}):
                op7 = float(operating[max(0, end + 1 - 7 * STEPS_PER_DAY) : end + 1].mean())
                op24 = float(operating[max(0, end + 1 - STEPS_PER_DAY) : end + 1].mean())
                packs.append(
                    {
                        "sample_id": f"{key}@{end}",
                        "farm": farm,
                        "event_id": event_id,
                        "turbine": info["turbine"],
                        "end_step": end,
                        "hours_to_window_end": round((w1 - end) / STEPS_PER_HOUR, 1),
                        "operating_7d": round(op7, 3),
                        "operating_24h": round(op24, 3),
                        "columns": {
                            column: describe(source, sensors, column)
                            | {"in_model": column in in_model}
                            | baseline_reference(raws[column], fit_rows)
                            | raw_window_stats(raws[column], valid, end)
                            | window_stats(z, end)
                            for column, z in sorted(zs.items())
                        },
                    }
                )
            print(f"{key} {info['label']:<7} columns={len(zs):>3} snapshots={args.per_event}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packs, indent=1))
    print(f"\n{len(packs)} snapshot packs -> {args.out}")


if __name__ == "__main__":
    main()
