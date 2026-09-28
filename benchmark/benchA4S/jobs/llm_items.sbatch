#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a4s_llm
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/llm_items.log
# One allocation serves every model call the benchA4S items need. Qwen2.5-32B comes up once and
# citation, then the Argument_Graph sentence labels, run against it, because a second allocation
# would spend its first ten minutes loading the same weights again. A paper's sentences are labelled
# independently, so the request width is raised to keep the server busy.
set -x
export AG_LABEL_WORKERS=32
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  EP=$(cat artifact-ai2science/_llm/llm_endpoint.txt 2>/dev/null)
  curl -s -m 3 "${EP}/models" >/dev/null && break
  sleep 15
done
echo "[wait] server up at $EP after $i polls"
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
export AG_LABEL_WORKERS=32
python3 scripts/run_slopA4S.py --checker citation
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage labels --runs 3"
kill $SERVER 2>/dev/null || true
