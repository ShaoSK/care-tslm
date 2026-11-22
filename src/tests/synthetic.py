"""A synthetic 'Wind Farm A' in the release's file format, with injected faults, for pipeline tests.

Component temperatures follow ambient temperature and load like real ones. A fault adds a slow
warming ramp to one component group over the last days before the fault, starting before the
labelled window (as seen for the gearbox events of the real data).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from care_tslm.config import CHANNELS

STEP = timedelta(minutes=10)
TRAIN_DAYS = 120
PREDICTION_DAYS = 20
WINDOW_DAYS = 7

DESCRIPTIONS = {
    "sensor_0": ("Ambient temperature", "�C"),
    "wind_speed_3": ("Windspeed", "m/s"),
    "power_30": ("Grid power", "kW"),
    "sensor_18": ("Generator rpm", "rpm"),
    "sensor_5": ("Pitch angle", "�"),
    "sensor_11": ("Temperature in gearbox bearing on high speed shaft", "�C"),
    "sensor_12": ("Temperature oil in gearbox", "�C"),
    "sensor_13": ("Temperature in generator bearing 2 (Drive End)", "�C"),
    "sensor_14": ("Temperature in generator bearing 1 (Non-Drive End)", "�C"),
    "sensor_15": ("Temperature inside generator in stator windings phase 1", "�C"),
    "sensor_38": ("Temperature in HV transformer phase L1", "�C"),
    "sensor_41": ("Temperature oil in hydraulic group", "�C"),
    "sensor_43": ("Nacelle temperature", "�C"),
}
FAULT_SENSORS = {"Gearbox failure": ("sensor_11", "sensor_12"), "Generator bearing failure": ("sensor_13",)}

# (event_id, asset, label, description)
EVENTS = (
    (1, 0, "anomaly", "Gearbox failure"),
    (2, 1, "anomaly", "Generator bearing failure"),
    (3, 2, "normal", ""),
    (4, 3, "normal", ""),
)
START = datetime(2021, 1, 1)


def _ts(step: int) -> str:
    return (START + step * STEP).strftime("%Y-%m-%d %H:%M:%S")


def _event_series(rng: np.random.Generator, n: int, fault: str, fault_start: int, fault_end: int) -> dict[str, np.ndarray]:
    steps = np.arange(n)
    day = steps / 144
    ambient = 8 + 10 * np.sin(2 * np.pi * (day - 100) / 365) + 4 * np.sin(2 * np.pi * day) + rng.normal(0, 0.5, n)
    wind = np.clip(7 + np.cumsum(rng.normal(0, 0.15, n)) * 0.05 + 3 * np.sin(2 * np.pi * day / 3) + rng.normal(0, 0.6, n), 0, 25)
    power = np.clip(((wind - 3) / 9) ** 3, 0, 1) * 2000 * (wind > 3)
    load = power / 2000
    out = {"sensor_0": ambient, "wind_speed_3": wind, "power_30": power}
    out["sensor_18"] = 1000 + 700 * np.sqrt(load) + rng.normal(0, 10, n)
    out["sensor_5"] = np.where(wind > 12, (wind - 12) * 2.0, 0.0) + rng.normal(0, 0.3, n)
    for sensor, (a, b, c) in {
        "sensor_11": (40, 0.5, 25), "sensor_12": (45, 0.4, 15), "sensor_13": (30, 0.6, 20), "sensor_14": (28, 0.6, 15),
        "sensor_15": (35, 0.5, 50), "sensor_38": (40, 0.4, 30), "sensor_41": (35, 0.3, 5), "sensor_43": (20, 0.8, 5),
    }.items():
        out[sensor] = a + b * ambient + c * load + rng.normal(0, 1.0, n)
    if fault in FAULT_SENSORS:
        ramp = np.clip((steps - (fault_start - 3 * 144)) / (fault_end - fault_start + 3 * 144), 0, 1)
        for sensor in FAULT_SENSORS[fault]:
            out[sensor] = out[sensor] + 4.0 * ramp * (load > 0.05)  # +4 degC at the fault, under load
    return out


def write_farm(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    directory = root / "Wind Farm A"
    (directory / "datasets").mkdir(parents=True, exist_ok=True)
    sensors = [c.sensor for c in CHANNELS["A"]]
    lines = ["sensor_name;statistics_type;description;unit;is_angle;is_counter"]
    lines += [f"{s};average;{DESCRIPTIONS[s][0]};{DESCRIPTIONS[s][1]};{s == 'sensor_5'};False" for s in sensors]
    (directory / "feature_description.csv").write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    n_train, n_pred = TRAIN_DAYS * 144, PREDICTION_DAYS * 144
    n = n_train + n_pred
    info = ["asset;event_id;event_label;event_start;event_start_id;event_end;event_end_id;event_description"]
    for event_id, asset, label, description in EVENTS:
        window_end = n - 1
        window_start = window_end - WINDOW_DAYS * 144 + 1
        info.append(f"{asset};{event_id};{label};{_ts(window_start)};0;{_ts(window_end)};0;{description}")
        series = _event_series(rng, n, description, window_start, window_end)
        status = np.zeros(n, dtype=int)
        status[rng.random(n) < 0.02] = 2
        if label == "anomaly":  # farm A style synthetic statuses before the fault
            status[n - 3 * 144 :] = 4
        gap = slice(n_train + 300, n_train + 330)  # a 5-hour hole in the file
        keep = np.ones(n, dtype=bool)
        keep[gap] = False
        header = ";".join(["time_stamp", "asset_id", "id", "train_test", "status_type_id", *[f"{s}_avg" for s in sensors]])
        rows = [header]
        for row_id, step in enumerate(np.flatnonzero(keep)):
            phase = "train" if step < n_train else "prediction"
            values = ";".join(f"{series[s][step]:.2f}" for s in sensors)
            rows.append(f"{_ts(step)};{asset};{row_id};{phase};{status[step]};{values}")
        (directory / "datasets" / f"{event_id}.csv").write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
    (directory / "event_info.csv").write_text("\r\n".join(info) + "\r\n", encoding="utf-8")
    return directory
