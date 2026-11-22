#!/bin/bash
# Multi-fold evaluation of PLAN.md 8.12 (pre-registered): the `last` answer format, one fold per test
# turbine, greedy generation, no threshold. Run on the GPU machine, in tmux:
#
#     tmux new-session -d -s folds "bash ~/timenet-hackathon/src/scripts/run_folds.sh 2>&1 | tee -a ~/logs_folds.log"
#
# Resumable at every step, so after a crash or a VM restart just start it again:
#   ~/folds/<T>.done      the fold is finished (predicted and scored) -> skipped
#   model_slim.pt exists  training and slimming are done -> straight to prediction
#   best_model.pt exists  epoch 1 was saved before the stop -> slim it, then predict
# predict.py itself skips snapshots it already wrote. A fold that fails is written to
# ~/folds/<T>.failed and the loop moves on; delete that file to retry it.
#
# Each fold keeps only a slim checkpoint (3.4 GB instead of 13.9 GB, scripts/slim_checkpoint.py):
# the disk holds 13 slim ones but not 13 full ones. The full file is deleted only after the slim one
# matches a reference that was verified tensor by tensor.
set -u

ORDER=${ORDER:-"B-12 B-13 B-6 B-7 B-11 B-14 A-11 A-13 A-21 B-0 B-2 B-5 A-10"}
STORE=~/data/store_ab_v3
R=~/OpenTSLM/results/Llama_3_2_1B/OpenTSLMFlamingo
REF=$R/stage6_wind_cot_test_A-0_last/checkpoints/model_slim.pt   # verified in full by slim_all.sh
SRC=~/timenet-hackathon/src
TRAIN_TIMEOUT_MIN=${TRAIN_TIMEOUT_MIN:-180}
ENV='cd ~/OpenTSLM && source .venv/bin/activate && set -a && source ~/.env && set +a && export PYTHONPATH=$PWD:$PWD/src:$HOME/timenet-hackathon/src && export PYTORCH_ALLOC_CONF=expandable_segments:True'
EFD='cd ~/efd && source .venv/bin/activate && cd ~/timenet-hackathon && export CUDA_VISIBLE_DEVICES='
mkdir -p ~/folds ~/predictions/folds ~/results/folds

log() { echo "[folds $(date -u +%m-%dT%H:%M)] $*"; }
fail() { log "$1 FAILED: $2"; echo "$2" > ~/folds/$1.failed; }

score() {  # name predictions out
  bash -c "$EFD && python src/eval/care_score.py --predictions $2 --care-root ~/data/CARE_To_Compare \
    --out $3 --persist-window-hours 24 --persist-needed-hours 1" 2>&1 | grep -E "^(A|B)[0-9]|care_score" | sed "s/^/    [$1] /"
}

# A-0 is the development fold: its `last` run already exists with this exact configuration.
if [ ! -f ~/folds/A-0.done ] && [ -f ~/predictions/tslm_A-0_last.jsonl ]; then
  cp ~/predictions/tslm_A-0_last.jsonl ~/predictions/folds/tslm_A-0_last.jsonl
  score A-0 ~/predictions/folds/tslm_A-0_last.jsonl ~/results/folds/care_A-0_last.json && touch ~/folds/A-0.done
  log "A-0 reused from the development run"
fi

for T in $ORDER; do
  [ -f ~/folds/$T.done ] && { log "$T already done"; continue; }
  [ -f ~/folds/$T.failed ] && { log "$T failed before ($(cat ~/folds/$T.failed)); skipped"; continue; }
  CK=$R/stage6_wind_cot_test_${T}_last/checkpoints
  LOG=~/folds/train_$T.log
  PRED=~/predictions/folds/tslm_${T}_last.jsonl

  if [ ! -f $CK/model_slim.pt ]; then
    if [ ! -f $CK/best_model.pt ]; then
      log "$T training"
      tmux kill-session -t train_fold 2>/dev/null
      tmux new-session -d -s train_fold "$ENV && python $SRC/train/stage6_wind_cot.py --store $STORE --test-turbines $T \
        --epochs 3 --anomaly-stride-hours 4 --batch-size 2 --gradient-checkpointing \
        --answer-format last --field-weights none 2>&1 | tee $LOG"
      for _ in $(seq 1 $((TRAIN_TIMEOUT_MIN * 6))); do
        grep -aq "New best model saved" $LOG 2>/dev/null && [ -f $CK/best_model.pt ] && break
        grep -aqE "Traceback|OutOfMemoryError" $LOG 2>/dev/null && break
        tmux has-session -t train_fold 2>/dev/null || break
        sleep 10
      done
      tmux kill-session -t train_fold 2>/dev/null; sleep 10
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

  log "$T predicting"
  bash -c "$ENV && python $SRC/train/predict.py --store $STORE --test-turbines $T --checkpoint $CK/model_slim.pt \
    --out $PRED --batch-size 4 --max-new-tokens 320 --answer-format last" > ~/folds/predict_$T.log 2>&1
  expected=$(grep -oE "^[0-9]+ test snapshots" ~/folds/predict_$T.log | head -1 | cut -d' ' -f1)
  got=$(wc -l < $PRED 2>/dev/null || echo 0)
  if [ -z "$expected" ] || [ "$got" -lt "$expected" ]; then
    fail $T "predict: $got of ${expected:-?} rows; $(grep -E 'Error|Traceback' ~/folds/predict_$T.log | tail -1)"; continue
  fi
  log "$T predicted $got rows, $(grep -c '"pred": "anomaly"' $PRED) alarms"
  score $T $PRED ~/results/folds/care_${T}_last.json && touch ~/folds/$T.done
  df -h / | tail -1 | awk '{print "    disk free " $4}'
done
log "all folds handled: $(ls ~/folds/*.done 2>/dev/null | wc -l) done, $(ls ~/folds/*.failed 2>/dev/null | wc -l) failed"
