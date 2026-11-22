"""Same-condition residuals: how far each component sensor is from what the turbine normally reads
under the same weather and load (PLAN.md 5.2).

For each component sensor of one event, fit on the baseline (first half of the training period,
status 0, producing power):

    y ~ w0 + w1*ambient + w2*power + w3*power^2 + w4*wind

then z = (y - y_hat) / sd(baseline residual). z = +2 means "2 typical deviations warmer than this
turbine normally runs in these conditions". Steps where the turbine is not producing power get NaN:
the model is only valid under load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from care_tslm.config import CHANNELS, LOAD_BINS, MIN_BASELINE_POINTS, OPERATING_POWER, Z_CLIP, sensor_pairs
from care_tslm.events import EventData


@dataclass
class Residuals:
    """The model input series of one event, on its 10-minute grid."""

    operating: np.ndarray
    """bool per step: power is above OPERATING_POWER (the status code is not used here)."""
    power_fraction: np.ndarray
    """Power divided by the baseline's 99th percentile (about rated power)."""
    z: dict[str, np.ndarray]
    """Residual z per component sensor (float32, NaN where not operating or missing)."""
    fit_points: dict[str, int]
    """How many baseline steps each sensor's model used. 0: no model, z is all NaN."""
    pair_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per sensor pair (``sensor_13-sensor_14``), the baseline statistics of the DIFFERENCE of their
    residuals: ``mean``, ``std``. Two sensors of one component track each other when healthy, so a
    shift in this difference survives season, load and model bias (PLAN.md 8.3 finding 4)."""
    load_baseline: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    """Per sensor and power band (``low``/``mid``/``high``), the baseline residual ``mean`` and
    ``std``. A fault can show up as a changed load RESPONSE while the overall mean is unremarkable."""
    baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per component sensor, what it READS on the healthy baseline, in its own physical unit:
    ``mean``, ``std``, ``p05``, ``median``, ``p95``. This is the 'normal standard' the rationale
    compares against -- a residual in sigma hides the level, and on load-dependent channels a large
    baseline sigma hides a large physical shift (PLAN.md 6)."""


def _role(farm: str, role: str) -> str:
    return next(c.sensor for c in CHANNELS[farm] if c.role == role)


def compute(event: EventData) -> Residuals:
    """Fit the normal-behaviour model on the event's baseline and return residual z series."""
    ambient = event.channels[_role(event.farm, "ambient")].astype(np.float64)
    power = event.channels[_role(event.farm, "power")].astype(np.float64)
    wind = event.channels[_role(event.farm, "wind")].astype(np.float64)

    base = np.zeros(event.n_steps, dtype=bool)
    base[event.train_steps[0] : event.baseline_end] = True
    base &= event.status == 0

    rated = np.nanpercentile(power[base], 99) if np.isfinite(power[base]).any() else np.nan
    if not np.isfinite(rated) or rated <= 0:
        rated = np.nanmax(np.abs(power)) or 1.0
    power_fraction = power / rated
    operating = np.nan_to_num(power_fraction, nan=0.0) > OPERATING_POWER

    design = np.stack([np.ones_like(power), ambient, power_fraction, power_fraction**2, wind], axis=1)
    usable_inputs = np.isfinite(design).all(axis=1)
    fit_rows = base & operating & usable_inputs

    z: dict[str, np.ndarray] = {}
    fit_points: dict[str, int] = {}
    baseline: dict[str, dict[str, float]] = {}
    for channel in CHANNELS[event.farm]:
        if channel.role != "component":
            continue
        y = event.channels[channel.sensor].astype(np.float64)
        rows = fit_rows & np.isfinite(y)
        n = int(rows.sum())
        fit_points[channel.sensor] = n
        if n >= MIN_BASELINE_POINTS:
            healthy = y[rows]
            q = np.percentile(healthy, [5, 50, 95])
            baseline[channel.sensor] = {
                "mean": round(float(healthy.mean()), 2),
                "std": round(float(healthy.std()), 2),
                "p05": round(float(q[0]), 2),
                "median": round(float(q[1]), 2),
                "p95": round(float(q[2]), 2),
            }
        out = np.full(event.n_steps, np.nan, dtype=np.float32)
        if n >= MIN_BASELINE_POINTS:
            # Standardize the inputs so the least-squares solve is well conditioned.
            mu = design[rows].mean(axis=0)
            sd = design[rows].std(axis=0)
            mu[0], sd[0] = 0.0, 1.0
            sd[sd == 0] = 1.0
            x = (design - mu) / sd
            weights, *_ = np.linalg.lstsq(x[rows], y[rows], rcond=None)
            residual = y - x @ weights
            scale = float(np.std(residual[rows]))
            if scale > 0:
                valid = operating & usable_inputs & np.isfinite(y)
                out[valid] = np.clip(residual[valid] / scale, -Z_CLIP, Z_CLIP)
        z[channel.sensor] = out

    # Baseline of each pair's difference, and of each sensor's residual per power band.
    pair_baseline: dict[str, dict[str, float]] = {}
    for _, a_sensor, b_sensor in sensor_pairs(event.farm):
        diff = z[a_sensor] - z[b_sensor]
        rows = fit_rows & np.isfinite(diff)
        if int(rows.sum()) >= MIN_BASELINE_POINTS:
            d = diff[rows]
            pair_baseline[f"{a_sensor}-{b_sensor}"] = {
                "mean": round(float(d.mean()), 2),
                "std": round(float(d.std()), 2),
            }
    load_baseline: dict[str, dict[str, dict[str, float]]] = {}
    for sensor, series in z.items():
        per_band: dict[str, dict[str, float]] = {}
        for band, lo, hi in LOAD_BINS:
            rows = fit_rows & np.isfinite(series) & (power_fraction > lo) & (power_fraction <= hi)
            n = int(rows.sum())
            if n >= 100:
                v = series[rows]
                per_band[band] = {"mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2), "n": n}
        if per_band:
            load_baseline[sensor] = per_band

    return Residuals(
        operating=operating,
        power_fraction=power_fraction.astype(np.float32),
        z=z,
        fit_points=fit_points,
        baseline=baseline,
        pair_baseline=pair_baseline,
        load_baseline=load_baseline,
    )
