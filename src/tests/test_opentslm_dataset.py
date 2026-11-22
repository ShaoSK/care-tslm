"""CAREWindFaultCoTDataset on the synthetic farm, with the OpenTSLM stub."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.build import build_store  # noqa: E402
from care_tslm.config import LONG_POINTS, SHORT_POINTS  # noqa: E402
from care_tslm.load_raw import load_farm  # noqa: E402
from care_tslm.opentslm_dataset import FoldSpec, build_split_rows, make_dataset_class, parse_answer  # noqa: E402
from tests import opentslm_stub  # noqa: E402
from tests.synthetic import write_farm  # noqa: E402

EOS = "<|eot|>"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("care_ds")
    write_farm(root)
    out = root / "store"
    build_store(load_farm(root, "A"), out, log=lambda *_: None)
    return out


def test_split_rows_respect_the_fold(store: Path) -> None:
    train, val, test, summary = build_split_rows(FoldSpec(store, ("A-0",), val_turbine=None))
    assert all(r["turbine"] != "A-0" for r in train + val)
    assert all(r["turbine"] == "A-0" for r in test)
    assert summary["train"]["anomaly"] > 0 and summary["train"]["normal"] > 0
    assert summary["test"]["n"] == 20 * 24  # hourly, the whole prediction period


def test_auto_validation_turbine_is_held_out(store: Path) -> None:
    train, val, _, summary = build_split_rows(FoldSpec(store, ("A-0",)))
    assert summary["val_turbine"] == "A-1"  # the other turbine with an anomaly event
    assert {r["turbine"] for r in val} == {"A-1"}
    assert all(r["turbine"] not in ("A-0", "A-1") for r in train)


def test_dataset_samples_have_the_opentslm_shape(store: Path) -> None:
    opentslm_stub.install()
    cls = make_dataset_class(FoldSpec(store, ("A-0",), val_turbine=None, max_test=10))
    train = cls("train", EOS_TOKEN=EOS)
    test = cls("test", EOS_TOKEN=EOS)
    assert len(test) == 10
    sample = train[0]
    assert sample["post_prompt"] == "Rationale:"
    # The verdict leads the answer and appears exactly once (care_tslm.text.render): a scoring run
    # can stop after a few tokens, and no trailing copy can contradict the leading one.
    assert sample["answer"].startswith(f"Answer: {sample['label']}\n")
    assert sample["answer"].count("Answer:") == 1
    assert parse_answer(sample["answer"]) == sample["label"]
    assert len(sample["time_series"]) == len(sample["time_series_text"]) == 26
    assert {len(s) for s in sample["time_series"]} == {LONG_POINTS} == {SHORT_POINTS}
    assert all(np.isfinite(s).all() for s in sample["time_series"])
    batch = opentslm_stub.collate_pad([dict(train[i]) for i in range(4)])
    assert batch[0]["time_series"].shape == (26, 180)  # 180 is a multiple of the patch size: no padding


def test_each_fold_gets_its_own_class(store: Path) -> None:
    opentslm_stub.install()
    a = make_dataset_class(FoldSpec(store, ("A-0",), val_turbine=None, max_test=5))
    b = make_dataset_class(FoldSpec(store, ("A-2",), val_turbine=None, max_test=5))
    assert {s["turbine"] for s in a("test", EOS_TOKEN=EOS)} == {"A-0"}
    assert {s["turbine"] for s in b("test", EOS_TOKEN=EOS)} == {"A-2"}


def test_parse_answer() -> None:
    assert parse_answer("... warmer than normal. Answer: anomaly") == "anomaly"
    assert parse_answer("Answer: normal. Answer: Anomaly") == "anomaly"
    assert parse_answer("no verdict") is None


def test_answer_only_trains_on_the_verdict_alone(store: Path) -> None:
    """PLAN.md 8.6: the pure-classification reference point -- no Component, no Evidence, and a
    prompt that asks for nothing else, so the whole loss falls on the verdict."""
    opentslm_stub.install()
    cls = make_dataset_class(FoldSpec(store, ("A-0",), val_turbine=None, answer_only=True, max_test=4))
    sample = cls("train", EOS_TOKEN=EOS)[0]
    assert sample["answer"] == f"Answer: {sample['label']}{EOS}"
    assert "Component:" not in sample["answer"] and "Evidence:" not in sample["answer"]
    assert "Reply with one line only" in sample["pre_prompt"]
    assert len(sample["time_series"]) == 26, "the curves are unchanged"
    assert parse_answer(sample["answer"]) == sample["label"]


def test_fields_format_reproduces_the_current_prompt_and_answer() -> None:
    """The prompt builder must give the 11-19 prompt byte for byte, or every v3w-style run changes."""
    from care_tslm import text

    assert text.pre_prompt("fields") == text.PRE_PROMPT
    fields = {"answer": "anomaly", "component": "gearbox", "evidence": "gearbox oil reads 71 °C"}
    assert text.render(fields) == "Answer: anomaly\nComponent: gearbox\nEvidence: gearbox oil reads 71 °C"
    assert text.render(fields, "fields") == text.render(fields)


def _sample(store: Path, answer_format: str) -> dict:
    opentslm_stub.install()
    spec = FoldSpec(store, ("A-0",), val_turbine=None, answer_format=answer_format, max_test=4)
    rows = make_dataset_class(spec)("train", EOS_TOKEN=EOS)
    # One sample of each label, so both assessment sentences are checked.
    by_label = {s["label"]: s for s in rows}
    assert set(by_label) == {"anomaly", "normal"}
    return by_label


def test_prefix_format_puts_a_fixed_assessment_before_the_verdict(store: Path) -> None:
    """PLAN.md 8.9 option A: one of v1's two closing sentences, then the verdict, then the fields."""
    from care_tslm import text

    for label, sample in _sample(store, "prefix").items():
        lines = sample["answer"].removesuffix(EOS).split("\n")
        assert lines[0] == f"Assessment: {text.ASSESSMENT[label]}"
        assert lines[1] == f"Answer: {label}"
        assert [line.split(":")[0] for line in lines] == ["Assessment", "Answer", "Component", "Evidence"]
        assert sample["answer"].count("Answer:") == 1
        assert parse_answer(sample["answer"]) == label
        assert "these four fields" in sample["pre_prompt"]
        assert sample["pre_prompt"].index("Assessment:") < sample["pre_prompt"].index("Answer:")
        assert len(sample["time_series"]) == 26, "the curves are unchanged"


def test_last_format_puts_the_verdict_after_the_evidence(store: Path) -> None:
    """PLAN.md 8.9 option B: the same three fields as the default, in reverse."""
    for label, sample in _sample(store, "last").items():
        answer = sample["answer"].removesuffix(EOS)
        assert answer.startswith("Evidence: ")
        assert answer.endswith(f"\nAnswer: {label}")
        assert [line.split(":")[0] for line in answer.split("\n")] == ["Evidence", "Component", "Answer"]
        assert parse_answer(answer) == label
        assert sample["pre_prompt"].index("Evidence:") < sample["pre_prompt"].index("Answer:")


def test_answer_formats_carry_the_same_fields(store: Path) -> None:
    """Only the order changes: the same rows get the same Component and Evidence text in all three."""
    def field_set(answer: str) -> set[str]:
        return {line for line in answer.removesuffix(EOS).split("\n") if not line.startswith("Assessment:")}

    by_format = {f: _sample(store, f) for f in ("fields", "prefix", "last")}
    for label in by_format["fields"]:
        reference = field_set(by_format["fields"][label]["answer"])
        assert field_set(by_format["prefix"][label]["answer"]) == reference
        assert field_set(by_format["last"][label]["answer"]) == reference


def test_answer_format_rejects_bad_combinations(store: Path) -> None:
    with pytest.raises(ValueError, match="not one of"):
        FoldSpec(store, ("A-0",), answer_format="verdict_last")
    with pytest.raises(ValueError, match="only applies"):
        FoldSpec(store, ("A-0",), answer_format="last", legacy_text=True)
    with pytest.raises(ValueError, match="only applies"):
        FoldSpec(store, ("A-0",), answer_format="prefix", answer_only=True)
    with pytest.raises(ValueError, match="exclusive"):
        FoldSpec(store, ("A-0",), legacy_text=True, answer_only=True)


def test_legacy_text_restores_the_v1_prompt_and_prose(store: Path) -> None:
    """PLAN.md 8.8: the 11-18 prompt with prose first and the verdict last, as v1 was trained."""
    from care_tslm import text

    opentslm_stub.install()
    rows = make_dataset_class(FoldSpec(store, ("A-0",), val_turbine=None, legacy_text=True, max_test=4))(
        "train", EOS_TOKEN=EOS
    )
    for sample in rows:
        answer = sample["answer"].removesuffix(EOS)
        assert sample["pre_prompt"] == text.PRE_PROMPT_LEGACY
        assert answer.startswith("Over the last 7 days the turbine produced power ")
        assert answer.endswith(f"Answer: {sample['label']}")
        assert "Component:" not in answer and "\n" not in answer


def test_verdict_candidates_are_the_start_of_the_trained_answer(store: Path) -> None:
    """Scoring must compare exactly what training taught: the candidate of the true label is the
    start of the target answer, and the two candidates differ before or at the verdict."""
    from care_tslm import text

    for fmt, answer_only in (("fields", False), ("prefix", False), ("fields", True)):
        opentslm_stub.install()
        spec = FoldSpec(store, ("A-0",), val_turbine=None, answer_format=fmt, answer_only=answer_only, max_test=4)
        candidates = text.verdict_candidates(fmt, answer_only=answer_only)
        assert set(candidates) == {"anomaly", "normal"}
        for sample in make_dataset_class(spec)("train", EOS_TOKEN=EOS):
            assert sample["answer"].startswith(candidates[sample["label"]])
            assert parse_answer(candidates[sample["label"]]) == sample["label"]
    with pytest.raises(ValueError, match="free text before the verdict"):
        text.verdict_candidates("last")


def test_before_verdict_cuts_the_models_answer_at_its_verdict(store: Path) -> None:
    """For ``last``: the trained answer cut before its verdict, plus the verdict, is the answer again."""
    from care_tslm import text

    opentslm_stub.install()
    spec = FoldSpec(store, ("A-0",), val_turbine=None, answer_format="last", max_test=4)
    for sample in make_dataset_class(spec)("train", EOS_TOKEN=EOS):
        answer = sample["answer"].removesuffix(EOS)
        own = text.before_verdict(answer)
        assert own is not None and own.startswith("Evidence: ")
        assert own + f"Answer: {sample['label']}" == answer
    assert text.before_verdict("no verdict here") is None
    assert text.before_verdict("Answer: normal ... Answer: anomaly") == "Answer: normal ... "
