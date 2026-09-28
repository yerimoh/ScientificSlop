#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a4s_b3a
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/rev_b3a.log
# Both reviewers that run on Qwen2.5-32B, in one allocation. They wait for the slop items' server to
# free rather than loading a second copy of the same weights beside it, and running them back to
# back means the 60GB of weights is read once for 988 reviews instead of twice.
set -x
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  EP=$(cat artifact-ai2science/_llm/llm_endpoint.txt 2>/dev/null)
  curl -s -m 3 "${EP}/models" >/dev/null && break
  sleep 15
done
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/bench_reviewsA4S.py --system b3a
python3 scripts/bench_reviewsA4S.py --system b3i
kill $SERVER 2>/dev/null || true
echo B3A_B3I_A4S_DONE
