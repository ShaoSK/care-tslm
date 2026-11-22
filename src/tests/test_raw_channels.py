"""The ``--with-raw`` build: component channels carried a second time as normalized raw readings."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm import text  # noqa: E402
from care_tslm.build import ArrayCache, build_store, load_events, load_meta  # noqa: E402
from care_tslm.config import CHANNELS, LONG_POINTS, base_sensor, channel_list  # noqa: E402
from care_tslm.load_raw import load_farm  # noqa: E402
from tests.synthetic import write_farm  # noqa: E402

N_COMPONENT = sum(c.role == "component" for c in CHANNELS["A"])


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("care")
    write_farm(root)
    plain, raw = root / "plain", root / "raw"
    build_store(load_farm(root, "A"), plain, log=lambda *_: None)
    build_store(load_farm(root, "A"), raw, with_raw=True, log=lambda *_: None)
    return plain, raw


def test_channel_list_appends_one_raw_copy_per_component():
    plain = channel_list("A")
    with_raw = channel_list("A", with_raw=True)
    assert list(plain) == list(CHANNELS["A"])
    assert len(with_raw) == len(plain) + N_COMPONENT
    added = with_raw[len(plain) :]
    assert all(c.role == "component_raw" for c in added)
    assert [base_sensor(c.sensor) for c in added] == [c.sensor for c in plain if c.role == "component"]


def test_the_store_grows_by_exactly_the_component_channels(stores):
    plain, raw = stores
    long_plain, _ = ArrayCache(plain).get(load_meta(plain)[0])
    long_raw, _ = ArrayCache(raw).get(load_meta(raw)[0])
    assert long_plain.shape == (len(CHANNELS["A"]), LONG_POINTS)
    assert long_raw.shape == (len(CHANNELS["A"]) + N_COMPONENT, LONG_POINTS)
    assert np.array_equal(
        np.nan_to_num(long_raw[: len(CHANNELS["A"])], nan=0.0),
        np.nan_to_num(long_plain, nan=0.0),
    ), "the residual channels must be untouched by the extra raw ones"


def test_the_raw_curve_holds_the_sensor_reading_not_the_residual(stores):
    _, raw = stores
    meta, events = load_meta(raw), load_events(raw)
    m = next(x for x in meta if x["role"] == "prediction")
    long, _ = ArrayCache(raw).get(m)
    names = events[f"{m['farm']}{m['event_id']}"]["channels"]
    i = names.index("sensor_12#raw")
    values = long[i][np.isfinite(long[i])]
    assert values.size, "the raw gearbox-oil curve should have data"
    assert values.mean() > 20, "a gearbox oil temperature in degrees, not a residual in sigma"


def test_the_rationale_rules_ignore_the_raw_channels(stores):
    """Thresholds and deviations must come from the residuals only, so v1 and v3 stay comparable."""
    plain, raw = stores
    t_plain = text.fit_thresholds(load_meta(plain))
    t_raw = text.fit_thresholds(load_meta(raw))
    assert t_plain == t_raw
    assert all("#raw" not in group for _, group, _ in t_raw)


def test_prompts_describe_the_raw_curves_as_normalized(stores):
    _, raw = stores
    meta, events = load_meta(raw), load_events(raw)
    m = meta[0]
    info = events[f"{m['farm']}{m['event_id']}"]
    long, short = ArrayCache(raw).get(m)
    prompts = text.channel_prompts("A", long, short, info["channel_text"], with_raw=True)
    assert len(prompts) == 2 * (len(CHANNELS["A"]) + N_COMPONENT)
    raw_prompts = [p for p in prompts if "raw reading" in p[0]]
    assert len(raw_prompts) == 2 * N_COMPONENT
    for caption, series in raw_prompts:
        assert "normalized" in caption and "sigma" not in caption
        assert np.isfinite(series).all(), "no NaN may reach the model"
        assert abs(float(series.mean())) < 0.5, "a per-window normalized curve is centred"


def test_a_plain_store_still_builds_without_raw_channels(stores):
    plain, _ = stores
    events = load_events(plain)
    assert all("component_raw" not in e["roles"] for e in events.values())
    assert all(not any("#raw" in c for c in e["channels"]) for e in events.values())
