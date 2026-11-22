"""The hourly -> 10-minute alarm expansion used for CARE-Score."""

from __future__ import annotations

import pandas as pd
import pytest

from care_tslm.scoring import STAMPS_PER_HOUR, alarm_series, naive_utc

GRID = pd.date_range("2016-03-01", periods=24 * 6, freq="10min")  # one day of 10-minute stamps


def row(ts: str, pred: str) -> dict:
    return {"end_us": int(pd.Timestamp(ts).value // 1000), "pred": pred}


def test_a_judgement_covers_the_hour_after_it_and_nothing_before():
    alarms, outside = alarm_series([row("2016-03-01 06:00", "anomaly")], GRID)
    assert outside == 0
    assert not alarms.loc[: pd.Timestamp("2016-03-01 05:50")].any(), "must not alarm before the snapshot ends"
    assert alarms.loc[pd.Timestamp("2016-03-01 06:00") : pd.Timestamp("2016-03-01 06:50")].all()
    assert not alarms.loc[pd.Timestamp("2016-03-01 07:00") :].any(), "a judgement expires after one hour"
    assert alarms.sum() == STAMPS_PER_HOUR


def test_consecutive_hours_join_into_one_continuous_alarm():
    rows = [row(f"2016-03-01 {h:02d}:00", "anomaly") for h in (6, 7, 8)]
    alarms, _ = alarm_series(rows, GRID)
    assert alarms.sum() == 3 * STAMPS_PER_HOUR
    assert alarms.loc[pd.Timestamp("2016-03-01 06:00") : pd.Timestamp("2016-03-01 08:50")].all()


def test_a_normal_judgement_clears_the_previous_alarm():
    rows = [row("2016-03-01 06:00", "anomaly"), row("2016-03-01 07:00", "normal")]
    alarms, _ = alarm_series(rows, GRID)
    assert alarms.loc[pd.Timestamp("2016-03-01 06:30")]
    assert not alarms.loc[pd.Timestamp("2016-03-01 07:30")]


def test_judgements_outside_the_prediction_period_are_reported_not_silently_used():
    rows = [row("2016-02-01 06:00", "anomaly"), row("2016-03-01 06:00", "anomaly")]
    alarms, outside = alarm_series(rows, GRID)
    assert outside == 1
    assert alarms.sum() == STAMPS_PER_HOUR


def test_no_predictions_means_no_alarms_rather_than_an_error():
    alarms, outside = alarm_series([], GRID)
    assert outside == 0 and not alarms.any() and len(alarms) == len(GRID)


def test_timezone_aware_index_is_compared_in_utc():
    aware = GRID.tz_localize("Europe/Berlin").tz_convert("UTC")
    assert naive_utc(aware).equals(GRID.tz_localize("Europe/Berlin").tz_convert("UTC").tz_localize(None))
    assert naive_utc(GRID).equals(GRID)


def test_a_judgement_for_a_stamp_the_file_skips_still_governs_the_stamps_after_it():
    """The CARE event files skip rows, so a judgement can land on a stamp that does not exist."""
    grid = GRID.drop(pd.Timestamp("2016-03-01 06:00"))
    alarms, outside = alarm_series([row("2016-03-01 06:00", "anomaly")], grid)
    assert outside == 0, "the judgement is inside the prediction period, just not on an existing row"
    assert alarms.loc[pd.Timestamp("2016-03-01 06:10") : pd.Timestamp("2016-03-01 06:50")].all()
    assert not alarms.loc[pd.Timestamp("2016-03-01 07:00") :].any()


@pytest.mark.parametrize("pred", ["normal", None])
def test_only_the_word_anomaly_raises_an_alarm(pred):
    alarms, _ = alarm_series([{"end_us": int(pd.Timestamp("2016-03-01 06:00").value // 1000), "pred": pred}], GRID)
    assert not alarms.any()


def test_threshold_rows_moves_the_decision_and_keeps_the_greedy_verdict() -> None:
    from care_tslm.scoring import threshold_rows

    rows = [{"p_anomaly": p, "pred": "normal"} for p in (0.05, 0.2, 0.6)]
    out = threshold_rows(rows, 0.1)
    assert [r["pred"] for r in out] == ["normal", "anomaly", "anomaly"]
    assert all(r["pred_greedy"] == "normal" for r in out)
    assert rows[1]["pred"] == "normal", "the input rows are not modified"


def test_rank_auc() -> None:
    from care_tslm.scoring import rank_auc

    assert rank_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert rank_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert rank_auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5  # ties count half
    assert rank_auc([0.1, 0.4, 0.35, 0.8], [False, True, False, True]) == 1.0
    assert rank_auc([0.3, 0.1, 0.2], [True, False, True]) == 1.0
    assert rank_auc([0.1, 0.2], [False, False]) is None
