"""The in-memory form of one CARE event, independent of how it was loaded."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from care_tslm.config import PERIOD_MIN

PERIOD_US = PERIOD_MIN * 60 * 1_000_000


@dataclass
class EventData:
    """One event file on its 10-minute grid.

    Step ``i`` is ``origin_us + i * PERIOD_US``. All step ranges are inclusive ``(first, last)``.
    """

    farm: str
    event_id: int
    asset_id: int
    label: str
    """``anomaly`` or ``normal``."""
    description: str | None
    origin_us: int
    """The absolute time of step 0, in microseconds since the epoch (anonymized years)."""
    train_steps: tuple[int, int]
    prediction_steps: tuple[int, int]
    window_steps: tuple[int, int]
    """The event window. For an anomaly event it ends where the turbine fault starts."""
    status: np.ndarray
    """int8 per step: the status ID, or -1 where the file has no row."""
    fault_component: str | None = None
    """The faulty component the operator's note names, from the release's label set
    (``hydraulic``, ``gearbox``, ``generator_bearing``, ``transformer``, ``rotor_bearing``).
    None for a normal event or a note that names no single component. This is ground truth, used as
    the rationale's ``Component`` field -- a training target, never an input."""
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    """float32 per step for each configured sensor, NaN where missing."""
    channel_text: dict[str, str] = field(default_factory=dict)
    """Each sensor's description with its unit, for the prompt."""

    @property
    def n_steps(self) -> int:
        return int(self.status.shape[0])

    @property
    def key(self) -> str:
        return f"{self.farm}{self.event_id}"

    @property
    def turbine(self) -> str:
        """The leave-one-turbine-out group key."""
        return f"{self.farm}-{self.asset_id}"

    @property
    def baseline_end(self) -> int:
        """The first step after the baseline: the first half of the training period (PLAN.md 5.1)."""
        first, last = self.train_steps
        return first + (last - first + 1) // 2

    def timestamp_us(self, step: int) -> int:
        return self.origin_us + step * PERIOD_US
