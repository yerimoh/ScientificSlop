#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a4s_revretry
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/rev_retry.log
# AI-Scientist and the CMU reviewer returned 494 empty reviews each. REVIEW_RAW keeps every model
# reply beside the review so the failure is readable instead of silent, and the first twenty items
# run alone before the rest so a repeat of the same failure is caught in minutes.
set -x
export REVIEW_RAW=1 REVIEW_WORKERS=8
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  EP=$(cat artifact-ai2science/_llm/llm_endpoint.txt 2>/dev/null)
  curl -s -m 3 "${EP}/models" >/dev/null && break
  sleep 15
done
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
rm -rf results/reviews/b3a results/reviews/b3i
python3 scripts/bench_reviewsA4S.py --system b3a
python3 scripts/bench_reviewsA4S.py --system b3i
kill $SERVER 2>/dev/null || true
echo REV_RETRY_DONE
