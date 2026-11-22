"""The residual-threshold baseline, end to end on the synthetic farm."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.build import build_store, load_meta  # noqa: E402
from care_tslm.load_raw import load_farm  # noqa: E402
from tests.synthetic import write_farm  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "residual_baseline.py"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("care_baseline")
    write_farm(root)
    out = root / "store"
    build_store(load_farm(root, "A"), out, log=lambda *_: None)
    return out


def run(store: Path, out: Path, *extra: str) -> list[dict]:
    turbine = sorted({m["turbine"] for m in load_meta(store)})[0]
    subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(store), "--test-turbines", turbine, "--out", str(out), *extra],
        check=True,
        capture_output=True,
    )
    with out.open() as f:
        return [json.loads(line) for line in f]


def test_predictions_have_the_same_fields_as_predict_py(store: Path, tmp_path: Path) -> None:
    rows = run(store, tmp_path / "baseline.jsonl")
    assert rows, "the baseline produced no predictions"
    required = {"farm", "event_id", "turbine", "end_step", "end_us", "status_end", "label", "pred", "generated"}
    assert required <= set(rows[0])
    assert {r["pred"] for r in rows} <= {"anomaly", "normal"}
    assert all("Answer: " in r["generated"] for r in rows), "the scorer parses the answer out of this field"


def test_it_only_scores_the_held_out_turbine(store: Path, tmp_path: Path) -> None:
    rows = run(store, tmp_path / "baseline.jsonl")
    assert len({r["turbine"] for r in rows}) == 1


def test_a_higher_threshold_cannot_produce_more_alarms(store: Path, tmp_path: Path) -> None:
    loose = run(store, tmp_path / "loose.jsonl", "--threshold-scale", "0.5")
    strict = run(store, tmp_path / "strict.jsonl", "--threshold-scale", "3.0")
    alarms = lambda rows: sum(r["pred"] == "anomaly" for r in rows)  # noqa: E731
    assert alarms(strict) <= alarms(loose)
