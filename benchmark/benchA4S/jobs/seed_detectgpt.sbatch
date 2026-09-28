#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-10:00:00
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/seed_dgpt_%j.log
# DetectGPT at one more seed. The T5 perturbations are sampled, so a single run is one draw of the
# score; the meeting asked for three draws and a mean. Four shards, one card each.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
DET=$ROOT/paper/draft_v6/claude/detectors
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python3 $DET/detectgpt.py --texts $B/scripts/texts.jsonl \
    --seed ${SEED} --shard $s --nshard 4 --out $B/results/seeds/detectgpt.s${SEED}.sh$s.jsonl &
done
wait
cat $B/results/seeds/detectgpt.s${SEED}.sh*.jsonl > $B/results/seeds/detectgpt.seed${SEED}.jsonl
echo DGPT_SEED_${SEED}_DONE
