#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-10:00:00
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/seed_dgpt165_%j.log
# DetectGPT at one more seed on the FARS half. The table pools the two halves, so the repeated runs
# have to cover both halves or the spread would describe only part of the benchmark.
set -x
ROOT=${SCISLOP_ROOT}
B165=$ROOT/paper/draft_v6/scislopbench/bench165
DET=$ROOT/paper/draft_v6/claude/detectors
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python3 $DET/detectgpt.py --texts $B165/scripts/texts.jsonl \
    --seed ${SEED} --shard $s --nshard 4 --out $B165/results/seeds/detectgpt.s${SEED}.sh$s.jsonl &
done
wait
cat $B165/results/seeds/detectgpt.s${SEED}.sh*.jsonl > $B165/results/seeds/detectgpt.seed${SEED}.jsonl
echo DGPT165_SEED_${SEED}_DONE
