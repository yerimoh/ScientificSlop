#!/bin/bash
# Fill the multi-year reviewer coverage once the 2026 run releases its GPUs, four workers per model.
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
while squeue -u $USER -h -o "%j" 2>/dev/null | grep -qE "rr_|r26|b3a_retry"; do sleep 300; done
S=reviewer_subset_years_fill.json
for s in 0 1 2 3; do
  q=$([ $s -lt 2 ] && echo ${SLURM_QOS} || echo ${SLURM_QOS})
  sbatch --partition=vram48 --qos=$q -N 1 -n 1 -c 8 --gres=gpu:1 --job-name=yf_b2h$s \
    -o logs/yfill_b2h_$s.log --wrap="bash scripts/revyr_worker.sh b2h $s 4 89$((10+s)) '' $S"
done
for s in 0 1 2 3; do
  q=$([ $s -lt 1 ] && echo ${SLURM_QOS} || echo q-low)
  sbatch --partition=vram48 --qos=$q -N 1 -n 1 -c 16 --gres=gpu:2 --job-name=yf_b3a$s \
    -o logs/yfill_b3a_$s.log --wrap="bash scripts/revyr_worker.sh b3a $s 4 89$((20+s)) '' $S"
done
echo "YEARS_FILL_SUBMITTED $(date '+%m-%d %H:%M')" >> logs/yfill.log
