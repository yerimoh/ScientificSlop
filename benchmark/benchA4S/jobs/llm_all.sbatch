#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=a4s_llm_all
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/llm_all.log
# Every Qwen2.5-32B call benchA4S needs, against the final views, in one allocation. The server is
# loaded once for the four runs instead of four times, and the slop items come first because the
# reviewers are the part that can be cut short if the allocation runs out.
set -x
export AG_LABEL_WORKERS=32 REVIEW_WORKERS=12
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  EP=$(cat artifact-ai2science/_llm/llm_endpoint.txt 2>/dev/null)
  curl -s -m 3 "${EP}/models" >/dev/null && break
  sleep 15
done
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/run_slopA4S.py --checker citation
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage labels --runs 3"
python3 scripts/bench_reviewsA4S.py --system b3a
python3 scripts/bench_reviewsA4S.py --system b3i
kill $SERVER 2>/dev/null || true
echo LLM_ALL_DONE
