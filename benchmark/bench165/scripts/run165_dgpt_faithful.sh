#!/bin/bash
# DetectGPT, faithful to Mitchell et al. 2023: t5-3b mask filling, 100 perturbations,
# scorer = GPT-J-6B (largest public model of the paper's source-model set; surrogate,
# since the true generator is closed). Usage: run165_dgpt_faithful.sh <shard> <nshard>
set -x
S=$1; N=$2
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
DET=$ROOT/paper/draft_v6/claude/detectors
LOCK=$B/results/supervisor/hf_dl.lock; mkdir -p $B/results/supervisor
flock $LOCK -c "hf download t5-3b >/dev/null 2>&1; hf download EleutherAI/gpt-j-6b >/dev/null 2>&1" || true
python3 $DET/detectgpt.py \
  --scorer EleutherAI/gpt-j-6b --perturber t5-3b --n_perturb 100 \
  --shard $S --nshard $N \
  --texts $B/scripts/texts.jsonl --out $B/results/detectgpt_faithful.s$S.jsonl
echo DGPT_FAITHFUL_S${S}_DONE
