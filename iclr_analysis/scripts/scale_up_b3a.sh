#!/bin/bash
# AI Scientist is the slow one (32B, two workers over 280 papers). As soon as the DetectGPT shards release their
# GPUs, add two reverse workers on the freed q-mid budget so the two ends meet in the middle. The runner skips a
# paper that already carries a score, so the only waste is the paper both ends reach at the same moment.
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
while squeue -u $USER -h -o "%j" 2>/dev/null | grep -q p26_d; do sleep 120; done
sleep 30
for s in 0 1; do
  sbatch --partition=vram48 --qos=${SLURM_QOS} -N 1 -n 1 -c 16 --gres=gpu:2 --job-name=r26b3aR$s \
    -o logs/rev26_b3a_r$s.log --wrap="bash scripts/rev26_worker.sh b3a $s 2 883$s --reverse"
done
echo "SCALED_UP $(date '+%m-%d %H:%M')" >> logs/scale_up.log
