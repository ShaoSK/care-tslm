Adds a connector for **CARE to Compare** (Fraunhofer IEE), a real-world benchmark for early fault detection in wind turbine SCADA data. It is the first energy/industrial dataset in the catalog.

## Source

- Zenodo record 15846963, version 6 (corrected labels): https://doi.org/10.5281/zenodo.15846963
- License: CC-BY-SA-4.0
- Paper: Gück, Roelofs & Faulstich (2024), *Data* 9(12):138, https://doi.org/10.3390/data9120138
- Archive pinned by SHA-256 (`ca61379e…94f1`), from a download whose MD5 matches the one Zenodo publishes (`2547b58c…500c`).

## What the dataset holds

- 95 Records, one per event file: 45 anomaly events (leading up to a documented turbine fault) and 50 normal events, across 36 turbines in 3 wind farms (A: 5 onshore, B: 9 and C: 22 offshore).
- Per Record: one Signal per sensor statistic at 10-minute resolution (A 81, B 252, C 952) on a `RegularAxis`, plus the turbine's operating status (`status_type_id`). Skipped grid steps and empty cells are nulls. Units map to pint where unambiguous; otherwise the release string is kept as a `source_unit` annotation.
- Annotations: wind farm, asset, event label, root-cause note, `training_period`, `prediction_period`, and the `event_window` interval.
- Tasks:
  - `ClassificationTask` (`event_label`: anomaly / normal) scoped to the prediction period, for every Record.
  - `TemporalLocalizationTask` targeting the `event_window` annotation, for every anomaly Record.
  - `ClassificationTask` (`fault_component`) for the 18 anomaly events of farms A and B, derived from the release's root-cause notes (documented on the card as TimeNet-derived).
- No split across events: the release defines a training and a prediction period inside each Record.

## Implementation notes

- `source.py` parses and validates the release: header, row IDs, the 10-minute grid, a single asset per file, training rows before prediction rows, and event windows on the grid. It has no TimeNet types, so it is tested on its own.
- Wind farm A names its eight energy counters (`sensor_44` to `sensor_51`) without a statistic suffix. They read as their sensor's single statistic. All three farms' headers were checked against their feature descriptions.
- `connector.py` caches each event's values as a feature-major `.npy` next to the extracted CSV and builds Signals with `Signal.from_loader`, so a build holds one event in memory at a time (wind farm C alone is about 17.8 GB of CSV).
- Tests use a synthetic wind farm in the release's shape (gaps, empty cells, a bare counter column, both asset column names, quoted descriptions) and include a storage round trip.

## Verification

- `pytest` on the connector tests: 13 passed. `ruff format`, `ruff check` and `ty check` are clean on the new package.
- Real data, wind farm A converted and written with `TimeFDataset.write`, then read back with `TimeFReader.verify()`: 22 Records × 82 Signals, 34 classification and 12 localization tasks. The labels (12 anomaly / 10 normal; 6 hydraulic, 3 gearbox, 2 generator bearing, 1 transformer) match the release's event table. Conversion took 76 s on a 2-core Codespace, and the written version is 84 MB.
- I have not run a full `timenet-build build fraunhofer-iee/care-to-compare` over all three farms. Wind farm C needs roughly 40 GB of free disk for extraction and cache.
