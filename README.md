# care-tslm — wind-turbine early fault warning with OpenTSLM

Team **Samosa**, TimeNet Online Hackathon, Round 1.

We fine-tune [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) (Flamingo + Llama-3.2-1B) to read a
wind turbine's recent SCADA curves every hour, say whether a fault is building up, and explain why —
and evaluate it with the official CARE-Score, leaving every turbine out in turn.

- **Data**: [CARE to Compare](https://doi.org/10.5281/zenodo.15846963) (Fraunhofer IEE, CC-BY-SA-4.0),
  wind farms A and B: 37 events (18 anomaly, 19 normal), 14 turbines, 31,842 hourly snapshots.
- **TimeNet connector** for the full release: [OpenTSLM/TimeNet#792](https://github.com/OpenTSLM/TimeNet/pull/792)
  (copy in [`connector/`](connector/)).
- **Demo**: [`docs/`](docs/) (GitHub Pages).

## Results

**Pre-registered leave-one-turbine-out evaluation**, 12 clean held-out folds (the two folds used to
choose settings are reported apart), alarm persistence (24 h, 1) for every method, one run per fold:

| | CARE-Score | anomaly events detected | normal events falsely detected |
|---|---|---|---|
| Fine-tuned OpenTSLM (`last` format, greedy) | 0.380 | 1 / 12 | 6 / 15 |
| Residual-threshold baseline | 0.477 | 7 / 12 | 12 / 15 |

The fine-tuned model **does not beat the baseline on unseen turbines**. Its development-fold score
(0.777) did not transfer: the same configuration scored 0.000 on the other development turbine.
CARE-Score can be gamed by hardly ever alarming (about 0.57 on these events), so the detected and
falsely detected counts belong next to it. Per-event table: `results/folds/summary.json`.

A **post-hoc follow-up (v4)** — positives only from the events with visible evidence, sensor-pair
difference curves as input, random residual offsets during training, and a per-turbine threshold set on
the turbine's own healthy training period — is reported separately in `results/v4/` and on the demo
page. It was designed after seeing the result above and does not replace it.

## What we learned

1. **Only 7 of the 18 anomaly events show evidence in these channels** (farm A 2/12, farm B 5/6); all six
   hydraulic events are invisible in 10-minute temperature means.
2. **Farm A's labels are confounded with season**: anomaly snapshots are 5–7 °C warmer on average.
3. **Sensor-pair differences** (two sensors of one component, e.g. main bearing 1 vs 2) are the most
   reliable evidence: they cancel season, load and normal-model bias.
4. **The model reads trends, not numbers**: numbers quoted in its explanations are off by ~21
   percentage points on average and must not be trusted.
5. **Greedy decoding hides a threshold**, and the order of the answer fields changes behaviour a lot
   (evidence before the verdict lets the model commit; verdict first makes it go silent).

## How it works

```
CARE CSV ──(connector parser)──> events ──> same-condition residuals ──> hourly multi-scale snapshots
   (30 d @ 4 h + 7.5 d @ 1 h, 180 points each; residual z for components, normalized context channels)
──> OpenTSLM-Flamingo (init: ECG-QA-CoT checkpoint) ──> "Evidence / Component / Answer"
──> hourly verdict or P(anomaly) ──> 10-min alarms ──> persistence filter ──> CARE-Score
```

- **Residuals** (`src/care_tslm/residuals.py`): per sensor, `y ~ ambient + power + power² + wind` fitted on
  the first half of each event's training year (status 0, producing power); that baseline never
  produces a sample.
- **Snapshots / labels** (`snapshots.py`): the label belongs to the snapshot's end time; anomaly-event
  hours outside the event window are scored but never trained on.
- **Folds** (`folds.py`, `opentslm_dataset.py`): by turbine; a training-side turbine is held out for
  early stopping; all text thresholds come from training-side turbines only.
- **Answers** (`text.py`): rule-generated `Evidence:` against the turbine's healthy baseline in physical
  units, `Component:` from the operator's root-cause label, `Answer: anomaly|normal`.
- **Scoring** (`train/score_answers.py`): P(anomaly) from the likelihood of the two candidate answers,
  instead of generating.
- **Evaluation** (`src/eval/care_score.py`): the official `CAREScore` from `energy-fault-detector`.

## Repository layout

| Path | What |
|---|---|
| `src/care_tslm/` | preprocessing, folds, prompt/answer text, OpenTSLM dataset, loss weighting, scoring helpers |
| `src/train/` | `stage6_wind_cot.py` (training, an OpenTSLM curriculum stage), `predict.py`, `score_answers.py` |
| `src/eval/` | CARE-Score evaluation, residual-threshold baseline |
| `src/scripts/` | sample building, multi-fold runners, summaries, calibration, checkpoint slimming |
| `src/tests/` | 107 tests on a synthetic wind farm with injected faults (`cd src && pytest tests`) |
| `connector/` | the TimeNet connector of PR #792 |
| `results/`, `predictions/` | CARE results and hourly predictions of every reported run |
| `docs/` | demo page |

## Reproduce

Two separate environments (the CARE evaluator pulls in TensorFlow, which conflicts with the training
stack):

```bash
# training / prediction: an OpenTSLM checkout, installed from its uv.lock, plus open-flamingo 2.0.1
cd OpenTSLM && uv sync --frozen && uv pip install open-flamingo==2.0.1 --no-deps
export PYTHONPATH=$PWD:$PWD/src:/path/to/care-tslm/src

python /path/to/care-tslm/src/scripts/build_samples.py --raw /data/CARE_To_Compare --farms A B --out data/store_ab_v3
python /path/to/care-tslm/src/train/stage6_wind_cot.py --store data/store_ab_v3 --test-turbines B-12 \
    --epochs 3 --anomaly-stride-hours 4 --batch-size 2 --gradient-checkpointing --answer-format last --field-weights none
# the full pre-registered run: src/scripts/run_folds.sh; the v4 follow-up: src/scripts/run_v4.sh

# evaluation: a second venv with energy-fault-detector
python src/scripts/summarize_folds.py --tslm 'predictions/folds/tslm_*_last.jsonl' \
    --baseline 'predictions/folds/baseline_*.jsonl' --care-root /data/CARE_To_Compare
```

Hardware: one 96 GB GPU (RTX PRO 6000); about 75 minutes of training per fold.

## Limitations

- One training run per fold; GPU non-determinism alone moved one fold's CARE-Score by 0.10.
- Farms A and B only; farm C is not used.
- Explanations are generated text; their numbers are unreliable (finding 4).
- The end-to-end pipeline reads the release through the connector's CSV parser; reading from a built
  TimeNet dataset (`src/care_tslm/load_timenet.py`) is implemented but was not verified on the full data.
- Some code comments refer to internal design notes that are not part of this repository.

## Credits

CARE to Compare: Gück, C., Roelofs, C., Faulstich, S. (Fraunhofer IEE), CC-BY-SA-4.0.
OpenTSLM: Stanford University, ETH Zurich and the OpenTSLM authors (MIT).
