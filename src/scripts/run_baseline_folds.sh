#!/bin/bash
# Residual-threshold baseline on every fold of PLAN.md 8.12, CPU only, same persistence (24, 1).
#     bash ~/timenet-hackathon/src/scripts/run_baseline_folds.sh 2>&1 | tee ~/logs_baseline_folds.log
set -u
TURBINES=${TURBINES:-"A-0 A-10 A-11 A-13 A-21 B-0 B-2 B-5 B-6 B-7 B-11 B-12 B-13 B-14"}
ENV='cd ~/OpenTSLM && source .venv/bin/activate && export PYTHONPATH=$PWD:$PWD/src:$HOME/timenet-hackathon/src && export CUDA_VISIBLE_DEVICES='
mkdir -p ~/predictions/folds ~/results/folds
for T in $TURBINES; do
  out=~/predictions/folds/baseline_$T.jsonl
  [ -s $out ] && { echo "[base] $T already done"; continue; }
  bash -c "$ENV && python ~/timenet-hackathon/src/eval/residual_baseline.py --store ~/data/store_ab_v3 \
    --test-turbines $T --out $out.tmp" > ~/folds_baseline_$T.log 2>&1 && mv $out.tmp $out \
    && echo "[base] $T $(wc -l < $out) rows, $(grep -c '"pred": "anomaly"' $out) alarms" \
    || echo "[base] $T FAILED: $(tail -1 ~/folds_baseline_$T.log)"
done
