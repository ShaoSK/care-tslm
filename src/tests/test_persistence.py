"""Alarm persistence: scattered hourly judgements -> sustained alarms (PLAN.md 8.1, fix 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from care_tslm.scoring import persist


def rows(pattern: str, start: str = "2016-03-01 00:00") -> list[dict]:
    """One hourly judgement per character: ``A`` anomaly, ``n`` normal, ``.`` no judgement at all."""
    out = []
    for hour, mark in enumerate(pattern):
        if mark == ".":
            continue
        ts = pd.Timestamp(start) + pd.Timedelta(hours=hour)
        out.append({"end_us": int(ts.value // 1000), "pred": "anomaly" if mark == "A" else "normal"})
    return out


def pattern_of(result: list[dict]) -> str:
    return "".join("A" if r["pred"] == "anomaly" else "n" for r in result)


def test_a_lone_pulse_is_suppressed():
    out = persist(rows("nnAnnnnn"), window_hours=4, needed_hours=2)
    assert pattern_of(out) == "nnnnnnnn", "one anomaly hour in four is below the 2-hour requirement"


def test_the_alarm_raises_once_the_requirement_is_met_and_holds_through_quiet_hours():
    out = persist(rows("nAnAnnnnnn"), window_hours=4, needed_hours=2)
    #              hours 1 and 3 are anomalies: at hour 3 and 4 the last 4 hours hold both of them
    assert pattern_of(out) == "nnnAAnnnnn", "the alarm holds while both hits stay inside the window"


def test_the_alarm_clears_when_the_hits_age_out_of_the_window():
    out = persist(rows("AAnnnnnn"), window_hours=3, needed_hours=2)
    assert pattern_of(out) == "nAAnnnnn", "hour 0 leaves the 3-hour window after hour 2"


def test_a_sustained_run_stays_on():
    out = persist(rows("nnAAAAAAn"), window_hours=4, needed_hours=2)
    assert pattern_of(out) == "nnnAAAAAA"


def test_window_one_needed_one_is_the_identity():
    original = rows("AnAAnnAn")
    out = persist(original, window_hours=1, needed_hours=1)
    assert pattern_of(out) == "AnAAnnAn"


def test_a_short_history_counts_what_it_has():
    out = persist(rows("AAnnnn"), window_hours=12, needed_hours=2)
    assert out[1]["pred"] == "anomaly", "two of the only two judgements so far were anomalies"


def test_a_gap_longer_than_the_window_does_not_carry_over():
    out = persist(rows("AA.........An"), window_hours=4, needed_hours=2)
    assert pattern_of(out) == "nAnn", "the late anomaly is alone in its own window"


def test_the_raw_judgement_is_kept_and_other_fields_survive():
    out = persist([{"end_us": 0, "pred": "anomaly", "turbine": "A-0"}], window_hours=4, needed_hours=2)
    assert out[0] == {"end_us": 0, "pred": "normal", "pred_raw": "anomaly", "turbine": "A-0"}


def test_judgements_are_returned_in_time_order_whatever_order_they_arrive_in():
    out = persist(rows("AAnnnn")[::-1], window_hours=3, needed_hours=2)
    assert [r["end_us"] for r in out] == sorted(r["end_us"] for r in out)
    assert pattern_of(out) == "nAAnnn"


def test_the_future_never_influences_an_earlier_hour():
    quiet = persist(rows("Annnn"), window_hours=4, needed_hours=2)
    later = persist(rows("AnnnnAAAA"), window_hours=4, needed_hours=2)
    assert pattern_of(later)[:5] == pattern_of(quiet), "adding later anomalies must not change earlier hours"


@pytest.mark.parametrize("window,needed", [(0, 1), (4, 0), (2, 3)])
def test_nonsensical_parameters_are_rejected(window, needed):
    with pytest.raises(ValueError):
        persist(rows("AA"), window_hours=window, needed_hours=needed)


def test_no_judgements_is_not_an_error():
    assert persist([], window_hours=4, needed_hours=2) == []
