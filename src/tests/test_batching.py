"""Batches must never mix wind farms (their samples carry different numbers of curves)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_tslm.batching import SameFarmBatchSampler  # noqa: E402

FARMS = ["A"] * 10 + ["B"] * 7  # 17 samples, an uneven split


def test_no_batch_mixes_farms():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4)
    for batch in sampler:
        assert len({FARMS[i] for i in batch}) == 1


def test_every_sample_is_used_exactly_once_per_epoch():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(len(FARMS)))


def test_len_matches_the_number_of_batches_the_scheduler_will_see():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4)
    assert len(sampler) == 3 + 2  # A: 4+4+2, B: 4+3
    assert len(list(iter(sampler))) == len(sampler)


def test_drop_last_drops_only_the_short_tail_of_each_farm():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4, drop_last=True)
    batches = list(iter(sampler))
    assert len(batches) == 2 + 1
    assert all(len(b) == 4 for b in batches)


def test_order_changes_between_epochs_but_the_content_does_not():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4, seed=1)
    first, second = list(iter(sampler)), list(iter(sampler))
    assert first != second, "each epoch should reshuffle"
    assert sorted(i for b in first for i in b) == sorted(i for b in second for i in b)


def test_shuffle_off_is_deterministic():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4, shuffle=False)
    assert list(iter(sampler)) == list(iter(sampler))


def test_a_single_farm_behaves_like_plain_batching():
    sampler = SameFarmBatchSampler(["A"] * 8, batch_size=4, shuffle=False)
    assert list(iter(sampler)) == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize("bad", [0, -1])
def test_a_meaningless_batch_size_is_rejected(bad):
    with pytest.raises(ValueError):
        SameFarmBatchSampler(FARMS, batch_size=bad)


def test_each_farm_can_get_its_own_batch_size():
    sampler = SameFarmBatchSampler(FARMS, batch_size=4, per_farm={"B": 1}, shuffle=False)
    sizes = {FARMS[b[0]]: len(b) for b in sampler}
    assert sizes["B"] == 1
    assert max(len(b) for b in sampler if FARMS[b[0]] == "A") == 4
    assert sorted(i for b in sampler for i in b) == list(range(len(FARMS)))


def test_the_heavier_farm_gets_a_smaller_batch():
    from care_tslm.batching import farm_batch_sizes

    sizes = farm_batch_sizes(2, {"A": 13, "B": 16})
    assert sizes == {"A": 2, "B": 1}, "B carries 16/13 as many curves, so it must not pair up"
    assert farm_batch_sizes(1, {"A": 13, "B": 16}) == {"A": 1, "B": 1}, "never below 1"
    assert farm_batch_sizes(4, {"A": 13}) == {"A": 4}, "a single farm keeps the asked-for size"


def test_a_zero_per_farm_batch_size_is_rejected():
    with pytest.raises(ValueError):
        SameFarmBatchSampler(FARMS, batch_size=4, per_farm={"B": 0})
