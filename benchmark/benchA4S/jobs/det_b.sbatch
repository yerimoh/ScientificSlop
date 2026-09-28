#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-10:00:00
#SBATCH --job-name=a4s_det_b
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/det_b.log
# DetectGPT is the one detector whose cost is per perturbation, 20 T5 rewrites of every scored
# window, so it is the only one worth sharding. Four shards, one GPU each, one output file each.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
DET=$ROOT/paper/draft_v6/claude/detectors
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python3 $DET/detectgpt.py --texts $B/scripts/texts.jsonl \
    --shard $s --nshard 4 --out $B/results/baselines/detectgpt.s$s.jsonl &
done
wait
cat $B/results/baselines/detectgpt.s*.jsonl > $B/results/baselines/detectgpt.jsonl
echo DET_B_DONE
