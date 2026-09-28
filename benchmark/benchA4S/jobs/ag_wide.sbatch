#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-16:00:00
#SBATCH --job-name=a4s_ag_wide
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/ag_wide.log
# Argument_Graph on the wider unit, introduction plus conclusion. The label stage needs the Qwen
# server and the PMI stage needs the GPUs the server is holding, so the server is brought down
# between them inside the one allocation rather than queueing twice for the same four cards.
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
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage labels --runs 3 --unit intro_concl"
kill $SERVER 2>/dev/null || true
sleep 60
export HF_HUB_OFFLINE=1
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage pmi --unit intro_concl"
echo AG_WIDE_DONE
