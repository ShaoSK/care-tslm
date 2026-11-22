"""Pair divergence and load response: the two features the teacher study found decisive (PLAN.md 8.3)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.config import sensor_pairs  # noqa: E402
from care_tslm.text import load_response, pair_divergence  # noqa: E402


def meta(pairs: dict | None = None, components: dict | None = None) -> dict:
    return {"farm": "A", "pairs": pairs or {}, "components": components or {}}


def test_pairs_are_formed_from_the_channel_table_never_hand_picked():
    pairs = sensor_pairs("A")
    assert ("gearbox", "sensor_11", "sensor_12") in pairs
    assert ("generator_bearing", "sensor_13", "sensor_14") in pairs
    assert all(g1 == g2 for g1, g2 in [(g, g) for g, _, _ in pairs]), "pairs never cross groups"
    b = sensor_pairs("B")
    assert ("transformer", "sensor_40", "sensor_47") in b, "transformer core vs cell"
    assert ("rotor_bearing", "sensor_51", "sensor_52") in b


def test_a_pair_that_still_tracks_is_not_reported():
    m = meta({"sensor_13-sensor_14": {"mean_7d": 0.10}})
    base = {"sensor_13-sensor_14": {"mean": 0.05, "std": 0.30}}
    (found,) = pair_divergence(m, base)
    assert abs(found["shift"]) < 1.0, "0.05 sigma against a 0.30 spread is nothing"


def test_a_diverged_pair_is_measured_in_its_own_healthy_spread():
    # B77: difference -3.53 against a healthy -0.04 with spread 0.34 -> about 10 sigma of the spread.
    m = meta({"sensor_13-sensor_14": {"mean_7d": -3.53}})
    base = {"sensor_13-sensor_14": {"mean": -0.04, "std": 0.34}}
    (found,) = pair_divergence(m, base)
    assert found["shift"] == pytest.approx((-3.53 + 0.04) / 0.34, rel=1e-6)
    assert abs(found["shift"]) > 10


def test_pairs_are_sorted_with_the_widest_divergence_first():
    m = meta({"sensor_11-sensor_12": {"mean_7d": 1.0}, "sensor_13-sensor_14": {"mean_7d": 3.0}})
    base = {"sensor_11-sensor_12": {"mean": 0.0, "std": 0.5}, "sensor_13-sensor_14": {"mean": 0.0, "std": 0.5}}
    assert [d["pair"] for d in pair_divergence(m, base)] == ["sensor_13-sensor_14", "sensor_11-sensor_12"]


@pytest.mark.parametrize("base", [None, {}, {"sensor_13-sensor_14": {"std": 0.0}}])
def test_a_pair_without_a_usable_baseline_is_skipped(base):
    assert pair_divergence(meta({"sensor_13-sensor_14": {"mean_7d": 5.0}}), base) == []


def test_load_response_reports_the_gap_between_high_and_low_power():
    # A45: +1.77 sigma above 80 % load against +0.51 in the reference, while below 35 % they agree.
    m = meta(components={"sensor_41": {"load_high": 1.77, "load_low": -0.30}})
    base = {"sensor_41": {"high": {"mean": 0.51}, "low": {"mean": -0.76}}}
    got = load_response(m, "sensor_41", base)
    assert got["high"] == (1.77, 0.51, pytest.approx(1.26))
    assert got["low"] == (-0.30, -0.76, pytest.approx(0.46))
    assert got["gap"] == pytest.approx(1.26 - 0.46)


def test_load_response_needs_the_high_band_and_one_other():
    m = meta(components={"sensor_41": {"load_high": 2.0}})
    assert load_response(m, "sensor_41", {"sensor_41": {"high": {"mean": 0.0}}}) is None, "one band is not a response"
    assert load_response(m, "sensor_41", None) is None


def test_a_sensor_that_responds_to_load_like_its_healthy_self_has_a_small_gap():
    m = meta(components={"sensor_41": {"load_high": -0.80, "load_low": 0.05}})
    base = {"sensor_41": {"high": {"mean": -0.91}, "low": {"mean": 0.00}}}
    assert abs(load_response(m, "sensor_41", base)["gap"]) < 0.2


def test_a_channel_running_cool_at_high_load_is_not_evidence():
    """PLAN.md 8.3 finding 3: a cool high-load residual is the ambient-extrapolation artefact, and
    it appears in turbines with and without the fault alike. Only extra heat counts."""
    from care_tslm.text import rationale_fields

    m = {
        "farm": "A",
        "target": "anomaly",
        "operating_7d": 0.9,
        "operating_24h": 0.9,
        "pairs": {},
        "components": {
            s: {"mean_30d": 0.0, "mean_7d": 0.0, "mean_24h": 0.0, "raw_mean_7d": 41.0, "load_high": v, "load_low": 0.4}
            for s, v in [("sensor_41", -0.88)]
        },
    }
    for sensor in ("sensor_18", "sensor_5", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_38", "sensor_43"):
        m["components"][sensor] = {"mean_30d": 0.0, "mean_7d": 0.0, "mean_24h": 0.0}
    load_base = {"sensor_41": {"high": {"mean": 0.0}, "low": {"mean": 0.0}}}
    fields, _ = rationale_fields(m, {"sensor_41": "Temperature oil in hydraulic group (°C)"}, {}, {}, "hydraulic", {}, {}, load_base)
    assert "load response" not in fields["evidence"], fields["evidence"]
