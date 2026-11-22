"""Batching that keeps one batch to one wind farm.

OpenTSLM's Flamingo collate stacks a sample's curves into a single tensor, so every sample in a
batch must carry the same *number* of curves. The farms have different sensor sets (A: 13 channels,
B: 16, each at two scales), so a mixed batch cannot be stacked. Grouping batches by farm keeps a
single model over both farms -- the point of a time-series language model -- without padding dummy
channels into the turbines that do not have those sensors.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import random


def farm_batch_sizes(batch_size: int, channels_per_farm: Mapping[str, int]) -> dict[str, int]:
    """Per-farm batch size, scaled down by how many curves that farm's samples carry.

    Peak activation memory grows with the number of curves in a batch, and the farms differ
    (A: 13 channels, B: 16, each at two scales). Giving every farm the same batch size makes the
    heavier farm set the memory ceiling: at ``--batch-size 2`` farm B pushed a 96 GB GPU to 86 GB.
    So the farm with the fewest channels keeps ``batch_size`` and the others are cut in proportion.
    """
    fewest = min(channels_per_farm.values())
    return {farm: max(1, int(batch_size * fewest / n)) for farm, n in channels_per_farm.items()}


class SameFarmBatchSampler:
    """Yields lists of dataset indices, each list from one farm only.

    The batch order is reshuffled every epoch (and the samples within each farm), so the model
    does not see the farms in a fixed rhythm.
    """

    def __init__(
        self,
        farms: Sequence[str],
        batch_size: int,
        *,
        per_farm: Mapping[str, int] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if per_farm and any(size < 1 for size in per_farm.values()):
            raise ValueError("every per-farm batch size must be at least 1")
        self.farms = list(farms)
        self.batch_size = batch_size
        self.per_farm = dict(per_farm or {})
        """Farms not listed here use ``batch_size`` (see :func:`farm_batch_sizes`)."""
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._groups: dict[str, list[int]] = {}
        for i, farm in enumerate(self.farms):
            self._groups.setdefault(farm, []).append(i)

    def _batches(self, rng: random.Random | None) -> list[list[int]]:
        batches = []
        for farm in sorted(self._groups):
            indices = list(self._groups[farm])
            size = self.per_farm.get(farm, self.batch_size)
            if rng is not None:
                rng.shuffle(indices)
            for start in range(0, len(indices), size):
                batch = indices[start : start + size]
                if len(batch) == size or not self.drop_last:
                    batches.append(batch)
        if rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random((self.seed, self.epoch).__hash__()) if self.shuffle else None
        self.epoch += 1
        return iter(self._batches(rng))

    def __len__(self) -> int:
        return len(self._batches(None))
