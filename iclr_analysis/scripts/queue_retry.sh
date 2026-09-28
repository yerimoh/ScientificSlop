#!/bin/bash
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
while squeue -u $USER -h -o "%j" 2>/dev/null | grep -q r26; do sleep 300; done
sleep 60
sbatch --partition=vram48 --qos=${SLURM_QOS} -N 1 -n 1 -c 16 --gres=gpu:2 --job-name=b3a_retry \
  -o logs/retry_b3a.log --wrap="bash scripts/retry_b3a_after.sh"
