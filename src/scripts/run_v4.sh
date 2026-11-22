#!/bin/bash
# v4 exploration of PLAN.md 8.13 (post hoc, after the pre-registered 8.12 result; it does NOT replace it):
#   1. evidence-only positives        --evidence-only   (config.EVIDENCE_EVENTS)
#   2. pair-difference input curves   store built with --with-pairs
#      + random residual offsets      --offset-aug 1.0  (training samples only)
#   3. per-turbine calibration        P(anomaly) by likelihood (prefix format), threshold = 95th
#                                     percentile of the test turbine's own training-period scores
# Run on the GPU machine, in tmux:
#
#     tmux new-session -d -s v4 "bash ~/timenet-hackathon/src/scripts/run_v4.sh 2>&1 | tee -a ~/logs_v4.log"
#
# Resumable like run_folds.sh: ~/v4/<T>.done skips a fold, a slim checkpoint skips training,
# score_answers.py skips rows it already wrote.
set -u

ORDER=${ORDER:-"B-13 B-11 B-12 A-13"}
STORE=~/data/store_ab_v4
R=~/OpenTSLM/results/Llama_3_2_1B/OpenTSLMFlamingo
REF=$R/stage6_wind_cot_test_A-0_last/checkpoints/model_slim.pt   # names and shapes only
SRC=~/timenet-hackathon/src
OUT=~/predictions/v4
TRAIN_TIMEOUT_MIN=${TRAIN_TIMEOUT_MIN:-180}
ENV='cd ~/OpenTSLM && source .venv/bin/activate && set -a && source ~/.env && set +a && export PYTHONPATH=$PWD:$PWD/src:$HOME/timenet-hackathon/src && export PYTORCH_ALLOC_CONF=expandable_segments:True'
EFD='cd ~/efd && source .venv/bin/activate && cd ~/timenet-hackathon && export CUDA_VISIBLE_DEVICES='
mkdir -p ~/v4 $OUT ~/results/v4

log() { echo "[v4 $(date -u +%m-%dT%H:%M)] $*"; }
fail() { log "$1 FAILED: $2"; echo "$2" > ~/v4/$1.failed; }

if [ ! -f $STORE/events.json ]; then
  log "building $STORE (pairs)"
  bash -c "$ENV && python $SRC/scripts/build_samples.py --raw ~/data/CARE_To_Compare --farms A B --out $STORE --with-pairs" \
    > ~/v4/build.log 2>&1 || { log "build FAILED: $(tail -1 ~/v4/build.log)"; exit 1; }
  log "built: $(grep -c . $STORE/samples.jsonl) snapshots"
fi

for T in $ORDER; do
  [ -f ~/v4/$T.done ] && { log "$T already done"; continue; }
  [ -f ~/v4/$T.failed ] && { log "$T failed before ($(cat ~/v4/$T.failed)); skipped"; continue; }
  CK=$R/stage6_wind_cot_test_${T}_prefix_evid_aug1/checkpoints
  LOG=~/v4/train_$T.log

  if [ ! -f $CK/model_slim.pt ]; then
    if [ ! -f $CK/best_model.pt ]; then
      log "$T training"
      tmux kill-session -t train_v4 2>/dev/null
      tmux new-session -d -s train_v4 "$ENV && python $SRC/train/stage6_wind_cot.py --store $STORE --test-turbines $T \
        --epochs 3 --anomaly-stride-hours 4 --batch-size 2 --gradient-checkpointing \
        --answer-format prefix --field-weights none --evidence-only --offset-aug 1.0 2>&1 | tee $LOG"
      for _ in $(seq 1 $((TRAIN_TIMEOUT_MIN * 6))); do
        grep -aq "New best model saved" $LOG 2>/dev/null && [ -f $CK/best_model.pt ] && break
        grep -aqE "Traceback|OutOfMemoryError" $LOG 2>/dev/null && break
        tmux has-session -t train_v4 2>/dev/null || break
        sleep 10
      done
      tmux kill-session -t train_v4 2>/dev/null; sleep 10
      if ! grep -aq "New best model saved" $LOG 2>/dev/null || [ ! -f $CK/best_model.pt ]; then
        fail $T "training: $(grep -aE 'Error|Traceback' $LOG | tail -1)"; continue
      fi
      log "$T epoch 1: $(grep -aE 'Epoch 1 . (train|val)' $LOG | tr '\n' ' ')"
    fi
    out=$(bash -c "$ENV && python $SRC/scripts/slim_checkpoint.py --checkpoint $CK/best_model.pt \
      --out $CK/model_slim.pt --reference $REF" 2>&1)
    if echo "$out" | grep -q "^MATCHES REFERENCE"; then
      rm -f $CK/best_model.pt; log "$T slimmed: $(echo "$out" | grep kept)"
    else
      rm -f $CK/model_slim.pt; fail $T "slim: $(echo "$out" | tail -1)"; continue
    fi
  fi

  for part in test calib; do
    extra=""; [ $part = calib ] && extra="--calibration --calibration-every 2"
    log "$T scoring $part"
    bash -c "$ENV && python $SRC/train/score_answers.py --store $STORE --test-turbines $T --checkpoint $CK/model_slim.pt \
      --out $OUT/${part}_$T.jsonl --batch-size 4 --answer-format prefix $extra" > ~/v4/score_${part}_$T.log 2>&1
    expected=$(grep -oE "^[0-9]+ test snapshots" ~/v4/score_${part}_$T.log | head -1 | cut -d' ' -f1)
    got=$(wc -l < $OUT/${part}_$T.jsonl 2>/dev/null || echo 0)
    if [ -z "$expected" ] || [ "$got" -lt "$expected" ]; then
      fail $T "score $part: $got of ${expected:-?}; $(grep -E 'Error|Traceback' ~/v4/score_${part}_$T.log | tail -1)"; continue 2
    fi
  done
  bash -c "$EFD && python src/scripts/calibrate_threshold.py --calib $OUT/calib_$T.jsonl --test $OUT/test_$T.jsonl \
    --out-dir $OUT" 2>&1 | sed "s/^/    /"
  touch ~/v4/$T.done
done
log "all v4 folds handled: $(ls ~/v4/*.done 2>/dev/null | wc -l) done, $(ls ~/v4/*.failed 2>/dev/null | wc -l) failed"

FOLDS=$(for T in $ORDER; do [ -f ~/v4/$T.done ] && echo -n "$T "; done)
[ -n "$FOLDS" ] && bash -c "$EFD && python src/scripts/compare_methods.py --care-root ~/data/CARE_To_Compare --folds $FOLDS \
  --method 'last=$HOME/predictions/folds/tslm_{T}_last.jsonl' \
  --method 'baseline=$HOME/predictions/folds/baseline_{T}.jsonl' \
  --method 'v4 greedy=$OUT/test_{T}.jsonl' \
  --method 'v4 cal0.9=$OUT/cal0.9_{T}.jsonl' \
  --method 'v4 cal0.95=$OUT/cal0.95_{T}.jsonl' \
  --method 'v4 cal0.99=$OUT/cal0.99_{T}.jsonl' \
  --out ~/results/v4/compare.json" 2>&1 | grep -v -i warn
