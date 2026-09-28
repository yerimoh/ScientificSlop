#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=a4s_llm2
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/llm_all2.log
# Every Qwen call benchA4S needs, against the views rebuilt after the word-spacing fix. Order puts
# the items first and the reviewers last, because a reviewer that fails is a baseline and an item
# that fails is the paper. REVIEW_RAW keeps the replies that the b3a and b3i failure hid.
set -x
export AG_LABEL_WORKERS=32 REVIEW_WORKERS=10 REVIEW_RAW=1
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  EP=$(cat artifact-ai2science/_llm/llm_endpoint.txt 2>/dev/null)
  curl -s -m 3 "${EP}/models" >/dev/null && break
  sleep 15
done
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
rm -rf results/reviews/b3a results/reviews/b3i results/reviews/b2h
python3 scripts/run_slopA4S.py --checker citation
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage labels --runs 3"
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage labels --runs 3 --unit intro_concl"
python3 scripts/bench_reviewsA4S.py --system b3a
python3 scripts/bench_reviewsA4S.py --system b3i
kill $SERVER 2>/dev/null || true
sleep 60
export HF_HUB_OFFLINE=1
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage pmi"
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage pmi --unit intro_concl"
echo LLM_ALL2_DONE
