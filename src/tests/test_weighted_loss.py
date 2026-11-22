"""Per-field loss weights: the verdict token must not be 1/40 of the gradient (PLAN.md 7)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.weighted_loss import field_token_weights  # noqa: E402

# A toy vocabulary: one id per word, so a "tokenization" is just a list of ids.
VOCAB = {w: i for i, w in enumerate(["<pad>", "\n", "Answer:", "Component:", "Evidence:", "anomaly", "normal", "gearbox", "a", "b", "c"])}


def ids(text: str) -> list[int]:
    return [VOCAB[w] for w in text.split(" ")]


def weights(answer: str, markers: dict[str, float], *, prompt: int = 3, default: float = 1.0):
    """Weights over a sequence of `prompt` prompt tokens followed by the answer."""
    seq = [VOCAB["a"]] * prompt + ids(answer)
    mask = np.array([False] * prompt + [True] * (len(seq) - prompt))
    marker_ids = {tuple(ids(m)): w for m, w in markers.items()}
    w = field_token_weights(np.array(seq), mask, marker_ids, newline_id=VOCAB["\n"], default=default)
    return w[prompt:]


def test_the_verdict_after_a_marker_is_upweighted():
    w = weights("Answer: anomaly \n a b", {"Answer:": 10.0})
    assert list(w) == [1.0, 10.0, 1.0, 1.0, 1.0], "only the token between the marker and the newline"


def test_every_occurrence_of_a_marker_is_weighted():
    w = weights("Answer: anomaly \n a \n Answer: anomaly", {"Answer:": 8.0})
    assert list(w) == [1.0, 8.0, 1.0, 1.0, 1.0, 1.0, 8.0]


def test_several_fields_get_their_own_weights():
    w = weights("Answer: anomaly \n Component: gearbox \n Evidence: a b c", {"Answer:": 9.0, "Component:": 4.0, "Evidence:": 0.5})
    assert list(w) == [1.0, 9.0, 1.0, 1.0, 4.0, 1.0, 1.0, 0.5, 0.5, 0.5]


def test_a_field_runs_to_the_end_when_no_newline_follows():
    w = weights("Evidence: a b c", {"Evidence:": 0.25})
    assert list(w) == [1.0, 0.25, 0.25, 0.25]


def test_the_marker_tokens_themselves_keep_the_default_weight():
    w = weights("Answer: anomaly", {"Answer:": 10.0})
    assert w[0] == 1.0, "the marker is boilerplate, not the decision"


def test_prompt_tokens_are_never_weighted():
    seq = np.array([VOCAB["Answer:"], VOCAB["anomaly"]] + ids("Answer: anomaly"))
    mask = np.array([False, False, True, True])
    w = field_token_weights(seq, mask, {tuple(ids("Answer:")): 10.0}, newline_id=VOCAB["\n"])
    assert list(w) == [0.0, 0.0, 1.0, 10.0], "masked positions carry no weight at all"


def test_a_marker_that_does_not_occur_changes_nothing():
    w = weights("a b c", {"Answer:": 10.0})
    assert list(w) == [1.0, 1.0, 1.0]


def test_all_weights_one_reproduces_plain_masking():
    w = weights("Answer: anomaly \n Evidence: a", {"Answer:": 1.0, "Evidence:": 1.0})
    assert list(w) == [1.0] * 5, "the identity case must be exactly the unweighted loss"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_nonpositive_weight_is_rejected(bad):
    with pytest.raises(ValueError):
        weights("Answer: anomaly", {"Answer:": bad})
