#!/bin/bash
# Binoculars, faithful to Hans et al. 2024: falcon-7b / falcon-7b-instruct,
# first 512-token window only (repo default max_token_observed=512), bf16 on A6000.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
DET=$ROOT/paper/draft_v6/claude/detectors
LOCK=$B/results/supervisor/hf_dl.lock; mkdir -p $B/results/supervisor
flock $LOCK -c "hf download tiiuae/falcon-7b >/dev/null 2>&1; hf download tiiuae/falcon-7b-instruct >/dev/null 2>&1" || true
python3 $DET/binoculars.py \
  --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct \
  --k 1 --window 512 \
  --texts $B/scripts/texts.jsonl --out $B/results/binoculars_faithful.jsonl
echo BINO_FAITHFUL_DONE
