"""Per-field loss weights for the answer text (PLAN.md 7).

The answer is trained with a plain language-modelling loss on its tokens, so the verdict -- the one
token that CARE-Score actually reads -- carries about 1/40 of the gradient while the rest goes into
reproducing the explanation's wording. These helpers give each named field of the answer its own
weight, so the classification decision can be made to matter without shortening the explanation.

Fields are located by their marker (``Answer:``, ``Component:``, ``Evidence:`` ...): a field runs
from the end of its marker to the next newline, or to the end of the answer. The marker tokens
themselves keep the default weight -- they are boilerplate, not the decision.

``make_weighted_compute_loss`` replaces ``OpenTSLMFlamingo.compute_loss``. The loss is normalized by
the sum of the weights rather than by the token count, so the gradient scale -- and therefore the
effective learning rate -- stays what it was with uniform weights. With every weight 1 the result is
exactly the unweighted loss, which ``tests/test_weighted_loss.py`` pins down.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "Answer:": 10.0,
    "Component:": 3.0,
}
"""Only the two decision fields are raised; prose keeps weight 1. ``text.render`` writes each of them
once; were a marker to appear twice, every occurrence would be weighted."""


def field_token_weights(
    token_ids: np.ndarray,
    answer_mask: np.ndarray,
    marker_ids: Mapping[Sequence[int], float],
    *,
    newline_id: int,
    default: float = 1.0,
) -> np.ndarray:
    """Weight per token of one sequence: ``default`` inside the answer, 0 outside, per-field where set.

    Args:
        token_ids: The whole sequence, prompt included.
        answer_mask: True where the token belongs to the answer (i.e. where the label is not -100).
        marker_ids: Each field marker as a tuple of token ids, mapped to the field's weight.
        newline_id: The token that ends a field.
        default: Weight of answer tokens no marker claims.

    Raises:
        ValueError: If a weight is not positive, or the inputs disagree in length.
    """
    if token_ids.shape != answer_mask.shape:
        raise ValueError(f"token_ids {token_ids.shape} and answer_mask {answer_mask.shape} differ")
    if any(w <= 0 for w in marker_ids.values()):
        raise ValueError(f"field weights must be positive, got {dict(marker_ids)}")
    weights = np.where(answer_mask, default, 0.0).astype(np.float32)
    n = len(token_ids)
    for marker, weight in marker_ids.items():
        marker = np.asarray(marker, dtype=token_ids.dtype)
        m = len(marker)
        if m == 0 or m > n:
            continue
        # Every start where the marker matches and the whole marker lies inside the answer.
        windows = np.lib.stride_tricks.sliding_window_view(token_ids, m)
        starts = np.flatnonzero((windows == marker).all(axis=1))
        for start in starts:
            field = start + m
            if field >= n or not answer_mask[start:field].all():
                continue
            end = field
            while end < n and answer_mask[end] and token_ids[end] != newline_id:
                end += 1
            weights[field:end] = weight
    return weights


def make_weighted_compute_loss(model, field_weights: Mapping[str, float]):
    """A ``compute_loss`` for ``OpenTSLMFlamingo`` that weights the answer's fields.

    Returns a function to bind onto the model instance. ``field_weights`` maps a field marker, as it
    appears in the answer text, to its weight.
    """
    import torch
    import torch.nn.functional as functional

    tokenizer = model.text_tokenizer
    marker_ids = {
        tuple(tokenizer.encode(marker, add_special_tokens=False)): float(weight)
        for marker, weight in field_weights.items()
    }
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    def compute_loss(self, batch):
        input_ids, images, attention_mask, labels = self.pad_and_apply_batch(batch, include_labels=False)
        output = self.model(vision_x=images, lang_x=input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits if hasattr(output, "logits") else output[1]

        ids_np = input_ids.detach().cpu().numpy()
        mask_np = (labels != -100).detach().cpu().numpy()
        weights = np.stack(
            [
                field_token_weights(ids_np[i], mask_np[i], marker_ids, newline_id=newline_id)
                for i in range(ids_np.shape[0])
            ]
        )
        weight = torch.from_numpy(weights).to(logits.device)

        # Next-token prediction: position t predicts token t + 1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_weight = weight[:, 1:] * (shift_labels != -100)
        token_loss = functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1).clamp(min=0),
            reduction="none",
        ).view(shift_labels.shape)
        total = shift_weight.sum()
        if total == 0:
            return output[0]
        return (token_loss * shift_weight).sum() / total

    return compute_loss.__get__(model, type(model))
