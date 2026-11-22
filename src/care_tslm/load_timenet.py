"""Load CARE events from a TimeNet dataset built by our connector (PR #792).

Produces the same :class:`EventData` as :mod:`care_tslm.load_raw`. ``scripts/check_loaders.py``
compares the two paths on real data; run it once TimeNet is installed (on the GPU machine).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from care_tslm.config import CHANNELS
from care_tslm.events import PERIOD_US, EventData


def _steps(span: object) -> tuple[int, int]:
    """Inclusive grid steps of a TimeInterval (``end_us`` is exclusive) on the record timeline."""
    start_us, end_us = int(span.start_us), int(span.end_us)  # type: ignore[attr-defined]
    return start_us // PERIOD_US, end_us // PERIOD_US - 1


def _signal_array(signal: object) -> np.ndarray:
    table = signal.to_arrow()  # type: ignore[attr-defined]
    column = table.column(table.num_columns - 1)  # the value column follows any time column
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def load_dataset(directory: Path, farms: tuple[str, ...] = ("A", "B")) -> Iterator[EventData]:
    """Yield every event of the given wind farms from a local TimeNet dataset version."""
    from timenet.reader import TimeFReader, open_local_version  # imported lazily: optional dependency

    reader = TimeFReader(open_local_version(directory))
    dataset = reader.read()
    for record in dataset.records:
        facts = {a.key: a for a in record.annotations}
        farm = str(facts["wind_farm"].value)
        if farm not in farms:
            continue
        signals = {s.name: s for source in record.sources for s in source.signals}
        status = _signal_array(signals["status_type_id"])
        status = np.where(np.isnan(status), -1, status).astype(np.int8)
        description = facts.get("event_description")
        data = EventData(
            farm=farm,
            event_id=int(facts["event_id"].value),
            asset_id=int(facts["asset_id"].value),
            label=str(facts["event_label"].value),
            description=None if description is None else str(description.value),
            origin_us=int(record.start_time.epoch_us),
            train_steps=_steps(facts["training_period"].span),
            prediction_steps=_steps(facts["prediction_period"].span),
            window_steps=_steps(facts["event_window"].span),
            status=status,
        )
        for channel in CHANNELS[farm]:
            signal = signals[channel.column]
            values = _signal_array(signal).astype(np.float32)
            if channel.zero_is_missing:
                values[values == 0] = np.nan
            data.channels[channel.sensor] = values
            # The connector names each spec "<description> (average)"; the unit is not repeated here.
            spec_name = getattr(getattr(signal, "spec", None), "name", None) or signal.name
            data.channel_text[channel.sensor] = str(spec_name).removesuffix(" (average)")
        yield data
