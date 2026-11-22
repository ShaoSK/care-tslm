"""Settings for the CARE early-fault-detection pipeline: channels, windows, sampling and thresholds.

Every number that shapes the samples lives here, so PLAN.md section 5 and the code stay in step.
"""

from __future__ import annotations

from dataclasses import dataclass

PERIOD_MIN = 10
"""The SCADA grid: one step every 10 minutes."""
STEPS_PER_HOUR = 60 // PERIOD_MIN
STEPS_PER_DAY = 24 * STEPS_PER_HOUR

# Multi-scale snapshot (PLAN.md 5.3).
# Both scales have 180 points: OpenTSLM's collate pads every series of a sample to the longest one
# (with zeros at the end, i.e. at the most recent time), so equal lengths avoid fake data.
LONG_DAYS = 30
LONG_BIN_HOURS = 4
SHORT_HOURS = 180  # 7.5 days
SHORT_BIN_HOURS = 1
LONG_STEPS = LONG_DAYS * STEPS_PER_DAY
SHORT_STEPS = SHORT_HOURS * STEPS_PER_HOUR
LONG_BIN = LONG_BIN_HOURS * STEPS_PER_HOUR
SHORT_BIN = SHORT_BIN_HOURS * STEPS_PER_HOUR
LONG_POINTS = LONG_STEPS // LONG_BIN  # 180
SHORT_POINTS = SHORT_STEPS // SHORT_BIN  # 180
LONG_LABEL = "last 30 days at 4-hour resolution"
SHORT_LABEL = "last 7.5 days at 1-hour resolution"

# Snapshot end-time strides (PLAN.md 5.5).
PREDICTION_STRIDE_HOURS = 1
"""Every hour of every prediction period gets a snapshot: these are scored with CARE."""
TRAIN_NORMAL_STRIDE_HOURS = 12
"""Normal snapshots from the second half of the training period are taken every 12 hours."""

# Normal-behaviour model (PLAN.md 5.2).
OPERATING_POWER = 0.05
"""A step counts as producing power above this fraction of rated power (power is scaled to rated)."""
MIN_BASELINE_POINTS = 500
"""A channel needs this many usable baseline steps to get a residual model."""
Z_CLIP = 8.0
"""Residual z values are clipped to +-Z_CLIP before they reach the model."""

# Rationale thresholds (PLAN.md 6), computed on the training fold's normal snapshots only.
THRESHOLD_QUANTILE = 0.99
MIN_THRESHOLD = 1.0


@dataclass(frozen=True)
class Channel:
    """One input channel: a SCADA sensor's 10-minute mean."""

    sensor: str
    """The anonymized sensor name in the release, such as ``sensor_12``."""
    role: str
    """``ambient``, ``power`` or ``wind`` for the operating-condition inputs, otherwise ``component``."""
    group: str
    """The component group the rationale names, such as ``gearbox``. Context channels use ``conditions``."""
    zero_is_missing: bool = False
    """Wind farms B and C were delivered with missing values replaced by 0. True for physical sensors
    where an exact 0 is not a plausible reading (temperatures, vibration)."""

    @property
    def column(self) -> str:
        """The event-file column that holds the 10-minute mean."""
        return f"{self.sensor}_avg"


def _component(sensor: str, group: str, *, zero_is_missing: bool = False) -> Channel:
    return Channel(sensor=sensor, role="component", group=group, zero_is_missing=zero_is_missing)


# PLAN.md 5.4. Descriptions and units come from the release's feature descriptions at load time.
CHANNELS: dict[str, tuple[Channel, ...]] = {
    "A": (
        Channel("sensor_0", "ambient", "conditions"),
        Channel("wind_speed_3", "wind", "conditions"),
        Channel("power_30", "power", "conditions"),
        _component("sensor_18", "drivetrain"),  # generator rpm
        _component("sensor_5", "pitch"),  # pitch angle
        _component("sensor_11", "gearbox"),  # gearbox bearing, high-speed shaft
        _component("sensor_12", "gearbox"),  # gearbox oil
        _component("sensor_13", "generator_bearing"),  # drive end
        _component("sensor_14", "generator_bearing"),  # non-drive end
        _component("sensor_15", "generator"),  # stator winding phase 1
        _component("sensor_38", "transformer"),  # HV transformer phase L1
        _component("sensor_41", "hydraulic"),  # hydraulic group oil
        _component("sensor_43", "nacelle"),  # nacelle temperature
    ),
    "B": (
        Channel("sensor_8", "ambient", "conditions", zero_is_missing=True),
        Channel("wind_speed_61", "wind", "conditions"),
        Channel("power_62", "power", "conditions"),
        _component("sensor_25", "drivetrain"),  # rotor speed
        _component("sensor_10", "pitch"),  # pitch angle
        _component("sensor_32", "generator_bearing", zero_is_missing=True),
        _component("sensor_33", "generator_bearing", zero_is_missing=True),
        _component("sensor_34", "gearbox", zero_is_missing=True),  # gearbox bearing 1
        _component("sensor_36", "gearbox", zero_is_missing=True),  # gearbox bearing 3
        _component("sensor_38", "gearbox", zero_is_missing=True),  # gearbox oil inlet
        _component("sensor_40", "transformer", zero_is_missing=True),  # transformer core
        _component("sensor_47", "transformer", zero_is_missing=True),  # transformer cell
        _component("sensor_51", "rotor_bearing", zero_is_missing=True),
        _component("sensor_52", "rotor_bearing", zero_is_missing=True),
        _component("sensor_53", "generator", zero_is_missing=True),  # slip ring
        _component("sensor_54", "vibration", zero_is_missing=True),  # drive train vibration Z
    ),
}

RAW_SUFFIX = "#raw"
"""Marks the second copy of a component channel: the sensor reading itself, not its residual."""


def base_sensor(name: str) -> str:
    """``sensor_12#raw`` -> ``sensor_12``; a plain name is returned unchanged."""
    return name.split(RAW_SUFFIX)[0]


PAIR_ROLE = "pair"
"""Role of a pair-difference channel: residual(a) - residual(b), in units of its healthy spread."""


def channel_list(farm: str, *, with_raw: bool = False, with_pairs: bool = False) -> tuple[Channel, ...]:
    """The channels a snapshot of this farm carries.

    With ``with_raw``, every component channel is carried a second time as the raw sensor reading
    (role ``component_raw``, normalized per window at prompt time) after all residual channels.
    The residual channels keep their positions, so a store built either way indexes the same.
    Rationale thresholds and deviations look at role ``component`` only, so the extra curves are
    input for the model and change nothing about the labels or the rules.

    With ``with_pairs``, every same-group sensor pair of :func:`sensor_pairs` is carried as one more
    channel (role ``pair``, sensor name ``a-b``), after everything else: the difference of the two
    residuals, centred and scaled by the healthy baseline's difference statistics. PLAN.md 8.3
    finding 4: the most reliable evidence, immune to season, load and normal-model bias.
    """
    channels = CHANNELS[farm]
    if with_raw:
        channels = channels + tuple(
            Channel(sensor=c.sensor + RAW_SUFFIX, role="component_raw", group=c.group, zero_is_missing=c.zero_is_missing)
            for c in CHANNELS[farm]
            if c.role == "component"
        )
    if with_pairs:
        channels = channels + tuple(Channel(sensor=f"{a}-{b}", role=PAIR_ROLE, group=g) for g, a, b in sensor_pairs(farm))
    return channels


LOAD_BINS: tuple[tuple[str, float, float], ...] = (
    ("low", OPERATING_POWER, 0.35),
    ("mid", 0.35, 0.80),
    ("high", 0.80, 10.0),
)
"""Power bands as a fraction of rated. The teacher study (PLAN.md 8.3) found one hydraulic fault
only because the oil's residual had acquired a load dependence the healthy machine lacks: +1.77 sigma
above 80 % load against +0.51 in the reference, while below 35 % the two agreed. A residual averaged
over all loads hides exactly that."""


def sensor_pairs(farm: str) -> tuple[tuple[str, str, str], ...]:
    """Every pair of component sensors inside the same group: (group, sensor_a, sensor_b).

    Two sensors of one component normally track each other, so the DIFFERENCE between their
    residuals cancels season, load and any bias in the normal-behaviour model. PLAN.md 8.3 finding 4:
    this was the only evidence that worked for four of the seven findable events (rotor bearing
    1 vs 2, transformer core vs cell). Pairs are formed generically from the channel table, never
    hand-picked per event -- picking per event would mean reading the test fold's answer.
    """
    groups: dict[str, list[str]] = {}
    for channel in CHANNELS[farm]:
        if channel.role == "component":
            groups.setdefault(channel.group, []).append(channel.sensor)
    pairs = []
    for group, sensors in groups.items():
        for i, a in enumerate(sensors):
            for b in sensors[i + 1 :]:
                pairs.append((group, a, b))
    return tuple(pairs)


EVIDENCE_EVENTS = frozenset({"A0", "A45", "B7", "B19", "B34", "B53", "B77"})
"""The anomaly events whose fault is visible in our SCADA channels (PLAN.md 8.4, two independent
studies). The other 11 show nothing before the fault -- training on their windows as 'anomaly' can
only teach a shortcut (season, turbine offset). With ``--evidence-only`` they are left out of the
training and validation positives; they are still scored. Only the training side's labels are used,
the test turbine's events are scored exactly as before."""


GROUP_NAMES = {
    "drivetrain": "drivetrain speed",
    "pitch": "pitch system",
    "gearbox": "gearbox",
    "generator_bearing": "generator bearings",
    "generator": "generator",
    "transformer": "transformer",
    "hydraulic": "hydraulic group",
    "nacelle": "nacelle",
    "rotor_bearing": "rotor (main) bearing",
    "vibration": "drive train vibration",
    "conditions": "operating conditions",
}
