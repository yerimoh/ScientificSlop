#!/bin/bash
# Everything the review paragraph still needs, submitted at once across the three queues.
# q-high and q-mid carry the work that must not be lost, q-low adds preemptible workers; a preempted worker
# only loses the paper it is on, because a record without a score is retried on the next pass.
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
sub() { # qos jobname gpus cpus wrap
  sbatch --partition=vram48 --qos=$1 -N 1 -n 1 -c $4 --gres=gpu:$3 --job-name=$2 -o logs/$2.log --wrap="$5"
}
# ICLR 2026, the 73 papers AI Scientist never reached
for s in 0 1; do
  sub ${SLURM_QOS} f26_b3a$s 2 16 "bash scripts/rev26_worker.sh b3a $s 2 891$s '' reviewer_subset_2026_fresh.json"
done
# the 13 whose review came back unparsable, at the raised output limit
sub q-low x26_b3a0 2 16 "B3A_MAX_TOKENS=6000 bash scripts/rev26_worker.sh b3a 0 1 8920 '' reviewer_subset_2026_failed.json"
# the multi-year fill, CycleReviewer over eight shards
for s in 0 1 2 3 4 5 6 7; do
  q=$([ $s -lt 4 ] && echo ${SLURM_QOS} || echo q-low)
  sub $q yf_b2h$s 1 8 "bash scripts/revyr_worker.sh b2h $s 8 893$s '' reviewer_subset_years_fill.json"
done
# the multi-year fill, AI Scientist over six shards
for s in 0 1 2 3 4 5; do
  q=$([ $s -lt 2 ] && echo ${SLURM_QOS} || echo q-low)
  sub $q yf_b3a$s 2 16 "bash scripts/revyr_worker.sh b3a $s 6 894$s '' reviewer_subset_years_fill.json"
done
echo "ALL_SUBMITTED $(date '+%m-%d %H:%M')" >> logs/yfill.log
