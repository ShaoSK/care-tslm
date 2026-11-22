"""Load CARE events straight from the release CSVs, with the parser of our TimeNet connector.

This is the development path: it needs only numpy, and it reads the files with the same code
(``source.py`` of PR #792) that the connector uses, so both paths give the same arrays.
"""

from __future__ import annotations

from collections.abc import Iterator
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

from care_tslm.config import CHANNELS, Channel
from care_tslm.events import EventData

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_PY = (
    PROJECT
    / "connector/packages/timenet-connectors/src/timenet_connectors/datasets/fraunhofer_iee/care_to_compare/source.py"
)


def _source_module() -> types.ModuleType:
    """Import the connector's ``source.py`` by path, without importing the rest of TimeNet."""
    name = "care_connector_source"
    if name in sys.modules:
        return sys.modules[name]
    try:
        import timenet.errors  # noqa: F401
    except ImportError:  # Only the error class is needed: stub it when TimeNet is not installed.
        errors = types.ModuleType("timenet.errors")
        errors.TimeNetConnectorError = type("TimeNetConnectorError", (Exception,), {})
        sys.modules.setdefault("timenet", types.ModuleType("timenet"))
        sys.modules["timenet.errors"] = errors
    import datetime

    if not hasattr(datetime, "UTC"):  # Python < 3.11 (TimeNet itself needs 3.11+)
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location(name, SOURCE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unit(text: str) -> str:
    return text.replace("�", "°").strip()


def channel_text(channel: Channel, description: str, unit: str) -> str:
    unit = _unit(unit)
    return f"{description} ({unit})" if unit else description


def farm_directory(root: Path, farm: str) -> Path:
    """The ``Wind Farm X`` directory under the extracted release (or a directory that is one)."""
    for candidate in (root / f"Wind Farm {farm}", root / "CARE_To_Compare" / f"Wind Farm {farm}"):
        if (candidate / "event_info.csv").exists():
            return candidate
    raise FileNotFoundError(f"no 'Wind Farm {farm}' with event_info.csv under {root}")


def load_farm(root: Path, farm: str, event_ids: list[int] | None = None) -> Iterator[EventData]:
    """Yield the events of one wind farm, in event ID order."""
    source = _source_module()
    directory = farm_directory(root, farm)
    sensors = source.read_sensor_descriptions(directory / "feature_description.csv")
    events = source.read_event_info(directory / "event_info.csv")
    channels = CHANNELS[farm]
    for event_id in sorted(events):
        if event_ids is not None and event_id not in event_ids:
            continue
        info = events[event_id]
        table = source.read_event_table(directory / "datasets" / f"{event_id}.csv")
        index = {column: i for i, column in enumerate(table.feature_columns)}
        data = EventData(
            farm=farm,
            event_id=event_id,
            asset_id=info.asset_id,
            label=info.label,
            description=info.description,
            fault_component=info.fault_component(farm),
            origin_us=table.start_us,
            train_steps=table.train_steps,
            prediction_steps=table.prediction_steps,
            window_steps=(table.step_of(info.start_us), table.step_of(info.end_us)),
            status=table.status,
        )
        for channel in channels:
            if channel.column not in index:
                raise KeyError(f"{farm}{event_id}: column {channel.column} is not in the event file")
            values = table.values[index[channel.column]].astype(np.float32, copy=True)
            if channel.zero_is_missing:
                values[values == 0] = np.nan
            data.channels[channel.sensor] = values
            sensor = sensors[channel.sensor]
            data.channel_text[channel.sensor] = channel_text(channel, sensor.description, sensor.unit)
        yield data
