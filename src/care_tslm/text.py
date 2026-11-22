"""Prompt and rule-based rationale text (PLAN.md 6).

The rationale is generated from the snapshot's own statistics, never by a language model, so every
number in it can be checked against the curves. The thresholds that decide "deviates from normal"
come from the training fold's normal snapshots only (no test information).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from care_tslm.config import (
    CHANNELS,
    LOAD_BINS,
    GROUP_NAMES,
    LONG_LABEL,
    MIN_THRESHOLD,
    PAIR_ROLE,
    SHORT_LABEL,
    THRESHOLD_QUANTILE,
    Z_CLIP,
    base_sensor,
    channel_list,
    sensor_pairs,
)

STATS = ("mean_30d", "mean_7d", "mean_24h")
STAT_TEXT = {"mean_30d": "over the last 30 days", "mean_7d": "over the last 7 days", "mean_24h": "over the last 24 hours"}

PRE_PROMPT = (
    "You are monitoring a wind turbine from its 10-minute SCADA data. For each component sensor you get "
    "its residual: how far it reads from what this turbine normally reads at the same ambient temperature, "
    "power and wind speed, in standard deviations (sigma) of normal operation. Each channel is given twice: "
    "the last 30 days at 4-hour resolution and the last 7.5 days at 1-hour resolution. Operating-condition "
    "channels (ambient temperature, power, wind speed) are given as normalized curves.\n"
    "Decide whether the turbine shows early signs of a developing fault. Possible answers: anomaly, normal.\n"
    "Reply in exactly these three fields, in this order:\n"
    "Answer: anomaly or normal\n"
    "Component: the faulty component area, or 'none' when the turbine is behaving normally\n"
    "Evidence: what each implicated channel reads now against this turbine's own healthy baseline, "
    "in its own unit, with the residual in sigma; 'unknown' when no channel deviates."
)
POST_PROMPT = "Rationale:"

_TASK = PRE_PROMPT[: PRE_PROMPT.index("Reply in exactly")]
"""The task description every field-based prompt shares, up to where the field list begins."""

ASSESSMENT = {
    "normal": "The turbine behaves as expected for its operating conditions.",
    "anomaly": "The signs are weak, but the overall pattern is treated as early abnormal behaviour.",
}
"""The two closing sentences of :func:`rationale` that v1 actually produced on fold A-0 (PLAN.md 8.8).

In v1 the verdict is a readout of this sentence (100 % conditional probability, both in v1 and in
its retrain). The ``prefix`` answer format puts it in front of the verdict to test whether this
short, learnable commitment alone is what lets the model alarm."""

_FIELD_TEXT = {
    "assessment": "Assessment: one sentence on whether the turbine behaves as expected or shows early abnormal "
    "behaviour",
    "answer": "Answer: anomaly or normal",
    "component": "Component: the faulty component area, or 'none' when the turbine is behaving normally",
    "evidence": "Evidence: what each implicated channel reads now against this turbine's own healthy baseline, "
    "in its own unit, with the residual in sigma; 'unknown' when no channel deviates.",
}

FIELD_ORDER = {
    "fields": ("answer", "component", "evidence"),
    "prefix": ("assessment", "answer", "component", "evidence"),
    "last": ("evidence", "component", "answer"),
}
"""Answer formats and the order of their fields.

``fields``: verdict first (the 11-19 rewrite). ``prefix``: a fixed one-sentence assessment before the
verdict (PLAN.md 8.9, option A). ``last``: verdict last, after the evidence (option B)."""

_COUNT = {3: "three", 4: "four"}


def pre_prompt(answer_format: str = "fields") -> str:
    """The prompt that asks for ``answer_format``; ``fields`` gives :data:`PRE_PROMPT` exactly."""
    order = FIELD_ORDER[answer_format]
    return (
        _TASK
        + f"Reply in exactly these {_COUNT[len(order)]} fields, in this order:\n"
        + "\n".join(_FIELD_TEXT[name] for name in order)
    )


PRE_PROMPT_LEGACY = (
    "You are monitoring a wind turbine from its 10-minute SCADA data. For each component sensor you get "
    "its residual: how far it reads from what this turbine normally reads at the same ambient temperature, "
    "power and wind speed, in standard deviations (sigma) of normal operation. Each channel is given twice: "
    "the last 30 days at 4-hour resolution and the last 7.5 days at 1-hour resolution. Operating-condition "
    "channels (ambient temperature, power, wind speed) are given as normalized curves.\n"
    "Decide whether the turbine shows early signs of a developing fault. Possible answers: anomaly, normal.\n"
    "Reason about the operating conditions and any component deviations, then end with 'Answer: anomaly' "
    "or 'Answer: normal'."
)
"""The prompt as it stood on 11-18, when the v1 run (CARE 0.7345) was trained.

It pairs with :func:`rationale` -- prose first, verdict last -- and was overwritten by the
three-field rewrite on 11-19. Recovered verbatim from the 11-18 session transcript; the project had
no git history at the time. Reachable through ``FoldSpec.legacy_text`` so the one configuration that
beat the residual baseline stays reproducible (PLAN.md 8.5)."""

PRE_PROMPT_ANSWER_ONLY = (
    "You are monitoring a wind turbine from its 10-minute SCADA data. For each component sensor you get "
    "its residual: how far it reads from what this turbine normally reads at the same ambient temperature, "
    "power and wind speed, in standard deviations (sigma) of normal operation. Each channel is given twice: "
    "the last 30 days at 4-hour resolution and the last 7.5 days at 1-hour resolution. Operating-condition "
    "channels (ambient temperature, power, wind speed) are given as normalized curves.\n"
    "Decide whether the turbine shows early signs of a developing fault. Reply with one line only:\n"
    "Answer: anomaly\n"
    "or\n"
    "Answer: normal"
)
"""The prompt for ``FoldSpec.answer_only``: no explanation is asked for, so none is promised.

With an answer of three tokens the whole loss falls on the verdict, which is the cleanest test of
whether the explanation text helps the classification or dilutes it (PLAN.md 8.6)."""


def _groups(farm: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for channel in CHANNELS[farm]:
        if channel.role == "component":
            groups[channel.group].append(channel.sensor)
    return dict(groups)


def group_scores(meta: dict) -> dict[str, dict[str, tuple[str, float]]]:
    """Per group and statistic, the sensor with the largest |value| and that value."""
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for group, sensors in _groups(meta["farm"]).items():
        per_stat = {}
        for stat in STATS:
            best: tuple[str, float] | None = None
            for sensor in sensors:
                value = meta["components"][sensor][stat]
                if value is not None and (best is None or abs(value) > abs(best[1])):
                    best = (sensor, value)
            if best is not None:
                per_stat[stat] = best
        out[group] = per_stat
    return out


Thresholds = dict[tuple[str, str, str], float]
"""(farm, group, statistic) -> |z| above which the group counts as deviating."""


def fit_thresholds(train_meta: Iterable[dict]) -> Thresholds:
    """High quantile of |group score| over the training fold's normal-pool snapshots."""
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for meta in train_meta:
        if meta["train_pool"] != "normal":
            continue
        for group, per_stat in group_scores(meta).items():
            for stat, (_, value) in per_stat.items():
                values[(meta["farm"], group, stat)].append(abs(value))
    return {
        key: max(MIN_THRESHOLD, float(np.quantile(v, THRESHOLD_QUANTILE))) for key, v in values.items() if v
    }


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def _sensor_name(text: str) -> str:
    """'Gearbox oil temperature (°C)' -> 'gearbox oil temperature'."""
    name = text.split(" (")[0].strip()
    return name[:1].lower() + name[1:]


def deviations(meta: dict, thresholds: Thresholds) -> list[dict]:
    """The groups whose score is above threshold, strongest first."""
    found = []
    for group, per_stat in group_scores(meta).items():
        hits = {}
        for stat, (sensor, value) in per_stat.items():
            limit = thresholds.get((meta["farm"], group, stat))
            if limit is not None and abs(value) > limit:
                hits[stat] = (sensor, value, limit)
        if hits:
            strength = max(abs(v) / lim for _, v, lim in hits.values())
            found.append({"group": group, "hits": hits, "strength": strength})
    return sorted(found, key=lambda d: -d["strength"])


FIELDS = ("answer", "component", "evidence")
"""The answer's fields, in the order they are written. ``src/train/stage6_wind_cot.py`` can weight
the loss per field, so each one must be a contiguous span of the answer text."""

NO_EVIDENCE = "unknown -- no component deviates from its healthy range for these conditions."
"""What the evidence field says when the rules find nothing. The label may still be ``anomaly``:
on this dataset three quarters of the in-window snapshots have no visible precursor (PLAN.md 6),
and saying so is the honest answer, not a guessed deviation."""

NO_COMPONENT = "none"
"""The component field of a snapshot labelled normal."""


def _physical(meta: dict, sensor: str, base: dict | None, channel_text: dict[str, str]) -> str | None:
    """``53.5 C vs healthy 50.6 (p95 57.0)`` -- the comparison a sigma value hides.

    Terse on purpose: every token of prose here is a token of gradient taken away from the verdict,
    and the numbers are what a reader checks against the curves.
    """
    stats = meta["components"].get(sensor) or {}
    now = stats.get("raw_mean_7d") or stats.get("raw_mean_30d")
    if now is None or not base:
        return None
    unit = channel_text.get(sensor, "")
    unit = unit[unit.rfind("(") + 1 : unit.rfind(")")].strip() if "(" in unit else ""
    piece = f"{now:g}{' ' + unit if unit else ''}, against {base['mean']:g} normally"
    if base.get("p95") is not None:
        piece += f" and {base['p95']:g} at its healthy peak"
    return piece


EXTRA_SIGNALS = ("trend", "spread", "excursion")
"""Signals beyond the residual level, from the teacher study (PLAN.md 6):

- ``trend``: the residual's least-squares slope in sigma per day over 30 days -- a component that
  is creeping warmer even while its mean still looks ordinary;
- ``spread``: this week's physical standard deviation over the healthy baseline's -- a bearing that
  runs normally most of the time and spikes intermittently, which no mean can show;
- ``excursion``: this week's estimated high reading (mean + 1.645 sd) against the healthy 95th
  percentile, in units of the baseline sd -- a level shift that a large baseline sigma hides.
"""

SIGNAL_TEXT = {
    "trend": "trending",
    "spread": "swinging wider than usual",
    "excursion": "reaching higher than its healthy range",
}


def _signal_values(meta: dict, sensor: str, base: dict | None) -> dict[str, float]:
    """The extra signals of one channel, or an empty dict when they cannot be computed."""
    stats = meta["components"].get(sensor) or {}
    out: dict[str, float] = {}
    slope = stats.get("slope_30d")
    if slope is not None:
        out["trend"] = float(slope)
    if not base or not base.get("std"):
        return out
    sd = base["std"]
    now_sd, now_mean = stats.get("raw_std_7d"), stats.get("raw_mean_7d")
    if now_sd is not None and sd > 0:
        out["spread"] = float(now_sd / sd)
    if now_mean is not None and now_sd is not None and base.get("p95") is not None and sd > 0:
        out["excursion"] = float((now_mean + 1.645 * now_sd - base["p95"]) / sd)
    return out


def group_signals(meta: dict, baseline: dict | None) -> dict[str, dict[str, tuple[str, float]]]:
    """Per group and extra signal, the channel with the largest value and that value."""
    baseline = baseline or {}
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for group, sensors in _groups(meta["farm"]).items():
        per_signal: dict[str, tuple[str, float]] = {}
        for sensor in sensors:
            for signal, value in _signal_values(meta, sensor, baseline.get(sensor)).items():
                best = per_signal.get(signal)
                if best is None or abs(value) > abs(best[1]):
                    per_signal[signal] = (sensor, value)
        out[group] = per_signal
    return out


def fit_signal_thresholds(train_meta: Iterable[dict], baselines: dict[str, dict]) -> Thresholds:
    """High quantile of each extra signal over the training fold's normal-pool snapshots.

    ``baselines`` maps an event key (``A0``) to that event's per-sensor baseline statistics. Fitted
    on the training turbines only, exactly like :func:`fit_thresholds` -- the test turbine's own
    distribution must never set the bar it is judged against.
    """
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for meta in train_meta:
        if meta["train_pool"] != "normal":
            continue
        base = baselines.get(f"{meta['farm']}{meta['event_id']}")
        for group, per_signal in group_signals(meta, base).items():
            for signal, (_, value) in per_signal.items():
                values[(meta["farm"], group, signal)].append(abs(value))
    return {key: float(np.quantile(v, THRESHOLD_QUANTILE)) for key, v in values.items() if v}


def extra_deviations(meta: dict, baseline: dict | None, thresholds: Thresholds) -> list[dict]:
    """The groups whose trend, spread or excursion is above threshold, strongest first."""
    found = []
    for group, per_signal in group_signals(meta, baseline).items():
        hits = {}
        for signal, (sensor, value) in per_signal.items():
            limit = thresholds.get((meta["farm"], group, signal))
            if limit is not None and limit > 0 and abs(value) > limit:
                hits[signal] = (sensor, value, limit)
        if hits:
            found.append(
                {
                    "group": group,
                    "hits": hits,
                    "strength": max(abs(v) / lim for _, v, lim in hits.values()),
                }
            )
    return sorted(found, key=lambda d: -d["strength"])


def pair_divergence(meta: dict, pair_baseline: dict | None) -> list[dict]:
    """Pairs whose residual difference has moved away from its own healthy value, strongest first.

    Two sensors of one component track each other when healthy, so a shift in their difference
    cannot be produced by season, load, or a drifting normal-behaviour model. Reported in units of
    the baseline spread of that difference, which is typically 0.2-0.4 sigma -- so a 1 sigma shift
    is already enormous (PLAN.md 8.3 finding 4).
    """
    pair_baseline = pair_baseline or {}
    found = []
    for group, a, b in sensor_pairs(meta["farm"]):
        name = f"{a}-{b}"
        stats, base = (meta.get("pairs") or {}).get(name), pair_baseline.get(name)
        if not stats or not base or not base.get("std"):
            continue
        now = stats.get("mean_7d")
        if now is None:
            continue
        shift = (now - base["mean"]) / base["std"]
        found.append(
            {"group": group, "pair": name, "sensors": (a, b), "now": now, "base": base, "shift": shift}
        )
    return sorted(found, key=lambda d: -abs(d["shift"]))


def load_response(meta: dict, sensor: str, load_baseline: dict | None) -> dict | None:
    """How a sensor's residual differs from its healthy value in the HIGHEST power band it has data
    for, and in the lowest -- a fault can change the load response while the overall mean is dull."""
    base = (load_baseline or {}).get(sensor)
    stats = meta["components"].get(sensor)
    if not base or not stats:
        return None
    bands = {}
    for band, _, _ in LOAD_BINS:
        now, ref = stats.get(f"load_{band}"), base.get(band)
        if now is not None and ref is not None:
            bands[band] = (now, ref["mean"], now - ref["mean"])
    if "high" not in bands or len(bands) < 2:
        return None
    low = bands.get("low") or bands.get("mid")
    return {"high": bands["high"], "low": low, "gap": bands["high"][2] - low[2]}


def rationale_fields(
    meta: dict,
    channel_text: dict[str, str],
    thresholds: Thresholds,
    baseline: dict | None = None,
    fault_component: str | None = None,
    signal_thresholds: Thresholds | None = None,
    pair_baseline: dict | None = None,
    load_baseline: dict | None = None,
) -> tuple[dict[str, str], dict]:
    """The answer as named fields plus facts about it (PLAN.md 6).

    Written as short labelled fields rather than prose. The evidence field leads with the channel's
    own reading against this turbine's healthy baseline in physical units, because a residual in
    sigma hides both the level and -- on load-dependent channels with a large baseline sigma -- the
    size of the physical shift. When nothing deviates the component field says ``unknown`` instead
    of naming a guess, and the note says plainly that the label, not the evidence, carries the alarm.
    """
    baseline = baseline or {}
    found = deviations(meta, thresholds)
    target = meta["target"]
    op7, op24 = meta["operating_7d"], meta["operating_24h"]

    extra = extra_deviations(meta, baseline, signal_thresholds or {})
    scores = group_scores(meta)

    def describe(sensor: str, value: float, signals: dict[str, tuple[str, float, float]]) -> str:
        """One clause: the channel's reading against this turbine's healthy baseline, its residual,
        and whichever of trend / spread / excursion also fired."""
        physical = _physical(meta, sensor, baseline.get(sensor), channel_text)
        name = _sensor_name(channel_text.get(sensor, sensor))
        head = (
            f"{name} reads {physical}, {_signed(value)} sigma"
            if physical
            else f"{name} is {_signed(value)} sigma from normal"
        )
        notes = []
        if "trend" in signals:
            notes.append(f"{SIGNAL_TEXT['trend']} {_signed(signals['trend'][1])} sigma per day over 30 days")
        if "spread" in signals:
            notes.append(f"{SIGNAL_TEXT['spread']} ({signals['spread'][1]:.1f}x its healthy spread)")
        if "excursion" in signals and not {"trend", "spread"} & set(signals):
            notes.append(SIGNAL_TEXT["excursion"])
        response = load_response(meta, sensor, load_baseline)
        # Only a channel running HOTTER than predicted counts: degradation adds heat. A residual that
        # is *cool* at high load is the ambient-extrapolation artefact of PLAN.md 8.3 finding 3 --
        # it shows up in five of six hydraulic-fault turbines and in turbines with no hydraulic fault
        # alike, because those snapshots sit 7-11 C warmer than the period the regression was fitted on.
        if response and response["gap"] > 1.0 and response["high"][0] > 0:
            hi, lo = response["high"], response["low"]
            notes.append(
                f"and its load response has changed: {_signed(hi[0])} sigma at high power "
                f"against {_signed(hi[1])} normally, while at low power it is {_signed(lo[0])} "
                f"against {_signed(lo[1])}"
            )
        return head + (", " + ", ".join(notes) if notes else "") + " for these conditions"

    by_group = {d["group"]: d for d in found}
    extra_by_group = {d["group"]: d for d in extra}
    # Fix A (PLAN.md 6): when the faulty component is known, report what IT reads, whether or not it
    # is the largest deviation. Its residual is above 1 sigma in 82 % of the in-window snapshots,
    # while the largest deviation lands on the faulty component in only 10 %.
    known = [fault_component] if target == "anomaly" and fault_component in scores else []
    order = known + [g for g in list(by_group) + list(extra_by_group) if g not in known]

    parts: list[str] = []
    for group in order[:2]:
        signals = extra_by_group[group]["hits"] if group in extra_by_group else {}
        level = by_group.get(group)
        if level:
            sensor, value, _ = max(level["hits"].values(), key=lambda hit: abs(hit[1]) / hit[2])
        else:
            scored = scores.get(group) or {}
            if not scored:
                continue
            sensor, value = max(scored.values(), key=lambda sv: abs(sv[1]))
        clause = describe(sensor, value, signals)
        # The caveat belongs to the clause it qualifies -- the documented component's own reading --
        # not to whichever channel happens to be described last.
        if group == fault_component and not level and not signals:
            clause += ", still inside its healthy range"
        parts.append(clause)
    # The most reliable feature first: two sensors of one component that no longer track each other.
    diverged = [d for d in pair_divergence(meta, pair_baseline) if abs(d["shift"]) > 3.0]
    for d in diverged[: 1 if parts else 2]:
        a_name = _sensor_name(channel_text.get(d["sensors"][0], d["sensors"][0]))
        b_name = _sensor_name(channel_text.get(d["sensors"][1], d["sensors"][1]))
        parts.insert(
            0,
            f"{a_name} and {b_name} no longer track each other: their difference is "
            f"{_signed(d['now'])} sigma against {_signed(d['base']['mean'])} when healthy "
            f"(healthy spread {d['base']['std']:.2f} sigma)",
        )
    evidence = "; ".join(parts) + "." if parts else NO_EVIDENCE

    if op24 < 0.2:
        evidence += f" Producing only {op24:.0%} of the last 24 hours, so recent points are few."

    # Ground truth from the operator's root-cause note, not a guess from the deviation rules --
    # those name the wrong component often enough to matter (A0's largest deviation is the
    # transformer, while the documented fault is the generator bearing).
    component = fault_component or NO_COMPONENT if target == "anomaly" else NO_COMPONENT
    fields = {"answer": target, "component": component, "evidence": evidence}
    facts = {
        "deviating_groups": [d["group"] for d in found],
        "extra_signal_groups": [d["group"] for d in extra],
        "diverged_pairs": [d["pair"] for d in pair_divergence(meta, pair_baseline) if abs(d["shift"]) > 3.0],
        "consistent": bool(found or extra) == (target == "anomaly"),
        "component": fields["component"],
        "evidence_supports_component": bool(found) and found[0]["group"] == fault_component,
    }
    return fields, facts


def render(fields: dict[str, str], answer_format: str = "fields") -> str:
    """The fields as the answer text the model is trained on, in the order of ``answer_format``.

    ``fields`` (the default) puts the verdict first; see :data:`FIELD_ORDER` for the other two.
    The verdict leads so that its first token is conditioned on the curves alone, not on an
    explanation the model has just written itself. It is NOT repeated at the end: the per-field loss
    weights of ``care_tslm.weighted_loss`` do what the repetition used to do, a trailing copy can
    silently contradict the leading one (``parse_answer`` reads the last match), and with the verdict
    first a scoring run can stop after a handful of tokens instead of generating the whole answer.
    """
    values = fields | {"assessment": ASSESSMENT[fields["answer"]]}
    return "\n".join(f"{name.capitalize()}: {values[name]}" for name in FIELD_ORDER[answer_format])


def verdict_candidates(answer_format: str = "fields", *, answer_only: bool = False) -> dict[str, str]:
    """For each label, the answer text up to and including the verdict -- what to score it by.

    Comparing the model's likelihood of the two candidates gives P(anomaly) without generating, so
    the decision threshold can be moved off the 50 % that greedy decoding implies (PLAN.md 8.10).
    Only for formats whose verdict comes before any free text: there the candidates differ in the
    verdict alone (or, with ``prefix``, in the fixed sentence that fixes it), and everything after it
    is conditioned on it rather than the other way round.

    Raises:
        ValueError: For ``last``, whose verdict follows text the model writes itself.
    """
    if answer_only or answer_format == "fields":
        return {label: f"Answer: {label}" for label in ASSESSMENT}
    if answer_format == "prefix":
        return {label: f"Assessment: {ASSESSMENT[label]}\nAnswer: {label}" for label in ASSESSMENT}
    raise ValueError(f"answer_format {answer_format!r} puts free text before the verdict; score it by generating")


def before_verdict(generated: str) -> str | None:
    """The model's own answer up to its last ``Answer:``, or None if it wrote none.

    For a format whose verdict follows text the model writes itself (``last``): appending
    ``Answer: anomaly`` and ``Answer: normal`` to this and comparing their likelihood gives
    P(anomaly | the model's own evidence) -- the probability at the verdict position, instead of
    only the token greedy decoding picked there (PLAN.md 8.11).
    """
    index = generated.rfind("Answer:")
    return None if index < 0 else generated[:index]


def rationale(meta: dict, channel_text: dict[str, str], thresholds: Thresholds) -> tuple[str, dict]:
    """The answer text for one snapshot and facts about it (for checks and the weak-sample ratio)."""
    parts = []
    op7, op24 = meta["operating_7d"], meta["operating_24h"]
    parts.append(f"Over the last 7 days the turbine produced power {op7:.0%} of the time ({op24:.0%} in the last 24 hours).")
    if op24 < 0.2:
        parts.append("In the last 24 hours it was mostly stopped or curtailed, so recent residuals are sparse.")

    found = deviations(meta, thresholds)
    for d in found[:3]:
        group_name = GROUP_NAMES[d["group"]]
        phrases = []
        for stat in STATS:
            if stat in d["hits"]:
                sensor, value, _ = d["hits"][stat]
                phrases.append(f"{_signed(value)} sigma {STAT_TEXT[stat]} ({_sensor_name(channel_text[sensor])})")
        limit = max(lim for _, _, lim in d["hits"].values())
        strongest = max(d["hits"].values(), key=lambda hit: abs(hit[1]) / hit[2])
        direction = "above" if strongest[1] > 0 else "below"
        parts.append(
            f"{group_name[:1].upper()}{group_name[1:]}: readings {direction} normal for these conditions, "
            f"{'; '.join(phrases)}, outside the usual range of about +-{limit:.1f} sigma."
        )
        persistent = "mean_30d" in d["hits"] or "mean_7d" in d["hits"]
        if not persistent:
            parts.append("This deviation is recent so far (last 24 hours only).")
    if not found:
        parts.append("No component deviates beyond its normal range after accounting for operating conditions.")

    target = meta["target"]
    if target == "anomaly" and found:
        parts.append(f"This points to developing abnormal behaviour, most likely in the {GROUP_NAMES[found[0]['group']]} area.")
    elif target == "anomaly":
        parts.append("The signs are weak, but the overall pattern is treated as early abnormal behaviour.")
    elif found:
        parts.append("Taken together with the other channels, this deviation alone is not treated as a developing fault.")
    else:
        parts.append("The turbine behaves as expected for its operating conditions.")
    parts.append(f"Answer: {target}")
    facts = {
        "deviating_groups": [d["group"] for d in found],
        "consistent": bool(found) == (target == "anomaly"),
    }
    return " ".join(parts), facts


def _curve_stats(curve: np.ndarray) -> tuple[float, float, float]:
    finite = curve[np.isfinite(curve)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0.0
    return float(finite.mean()), float(finite.std()), finite.size / curve.size


RAW_PROMPT_NOTE = (
    " Every component sensor is also given as its raw reading, normalized per window: the residual "
    "says how far from normal, the raw curve keeps the absolute level and shape."
)


PAIR_PROMPT_NOTE = (
    " Sensor pairs of one component are also given as the difference of their residuals, in units of "
    "that difference's healthy spread: two healthy sensors of one component track each other, so a "
    "difference that leaves 0 is not explained by weather, load or season."
)


def channel_prompts(
    farm: str,
    long: np.ndarray,
    short: np.ndarray,
    channel_text: dict[str, str],
    *,
    with_raw: bool = False,
    with_pairs: bool = False,
    offsets: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """(text, series) per channel and scale, in the order the model sees them.

    Component residuals are passed as-is (they are already in sigma units, clipped); missing bins
    become 0 (= normal). Context channels are normalized per window, with mean/std in the text, like
    OpenTSLM's HAR and ECG datasets. Pair differences (role ``pair``) are passed as-is too.

    ``offsets`` (one value per channel, training only) is added to every component residual before
    its text is written, on both scales alike: the augmentation of PLAN.md 8.13 that makes a constant
    level unreliable, so the model has to read shape and pair differences instead of 'this turbine
    runs warm'. Pair channels are never shifted.
    """
    out = []
    channels = channel_list(farm, with_raw=with_raw, with_pairs=with_pairs)
    for data, label in ((long, LONG_LABEL), (short, SHORT_LABEL)):
        for i, channel in enumerate(channels):
            curve = data[i].astype(np.float32)
            if offsets is not None and channel.role == "component" and offsets[i]:
                curve = np.clip(curve + np.float32(offsets[i]), -Z_CLIP, Z_CLIP)
            mean, std, coverage = _curve_stats(curve)
            missing = "" if coverage > 0.95 else f" {1 - coverage:.0%} of the bins have no data (shown as 0)."
            if channel.role == "component_raw":
                name = channel_text[base_sensor(channel.sensor)]
                if coverage == 0:
                    text, series = f"{name}, raw reading, {label}: no data.", np.zeros_like(curve)
                else:
                    scale_std = std if std > 1e-6 else 1.0
                    text = (
                        f"{name}, raw reading, {label}, normalized: mean {mean:.2f}, std {std:.2f}." + missing
                    )
                    series = np.nan_to_num((curve - mean) / scale_std, nan=0.0)
            elif channel.role == PAIR_ROLE:
                a, b = channel.sensor.split("-")
                name = f"{channel_text[a]} minus {channel_text[b]}"
                peak = float(np.nanmax(np.abs(curve))) if coverage > 0 else float("nan")
                text = (
                    f"{name}, difference of residuals vs. its healthy level, in healthy spreads, {label}. "
                    f"Mean {_signed(mean) if coverage else 'n/a'}, largest |value| {peak:.1f}." + missing
                )
                series = np.nan_to_num(curve, nan=0.0)
            elif channel.role == "component":
                name = channel_text[channel.sensor]
                peak = float(np.nanmax(np.abs(curve))) if coverage > 0 else float("nan")
                text = (
                    f"{name}, residual vs. normal operation in sigma, {label}. "
                    f"Mean {_signed(mean) if coverage else 'n/a'}, largest |value| {peak:.1f}." + missing
                )
                series = np.nan_to_num(curve, nan=0.0)
            else:
                name = "Power (fraction of rated)" if channel.role == "power" else channel_text[channel.sensor]
                if coverage == 0:
                    text, series = f"{name}, {label}: no data.", np.zeros_like(curve)
                else:
                    scale_std = std if std > 1e-6 else 1.0
                    text = f"{name}, {label}, normalized: mean {mean:.2f}, std {std:.2f}." + missing
                    series = np.nan_to_num((curve - mean) / scale_std, nan=0.0)
            out.append((text, series.astype(np.float32)))
    return out
