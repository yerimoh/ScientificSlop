#!/bin/bash
# finish_variant.sh TAG MODEL SUBSET  -- detached post-processing for a model variant.
# Waits until the four arms are complete for the subset, then measures, aggregates and draws.
TAG=$1; MODEL=$2; SUBSET=$3
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); cd "$ROOT"
export EOR_TAG="$TAG" EOR_MODEL="$MODEL"
LOG="$ROOT/logs/finish${TAG}.log"
while true; do
  left=0
  for a in a1_base a2_code a3_review a4_slop; do
    l=$(python3 code/remaining.py $a 3 "$SUBSET" 2>/dev/null || echo 999); left=$((left + l))
  done
  echo "[$(date +%m-%d\ %H:%M)] papers left across arms: $left" >> "$LOG"
  [ "$left" = "0" ] && break
  sleep 600
done
bash code/measure_all.sh >> "$LOG" 2>&1
python3 code/aggregate.py >> "$LOG" 2>&1
python3 - >> "$LOG" 2>&1 <<'PY'
import json, os
from pathlib import Path
tag=os.environ['EOR_TAG']; E=json.load(open(f'results/summary/effects{tag}.json'))
print(f'=== variant {tag} R3 means (n per item) ===')
for arm,A in E['arms'].items():
    R=A['rounds'].get('3') or A['rounds'].get(3)
    if R: print(arm, {it:(v['mean_rn'], v['n']) for it,v in R['items'].items()}, 'noop', R['n_failed_noop'])
PY
echo "[$(date +%m-%d\ %H:%M)] FINISHED variant $TAG" >> "$LOG"
