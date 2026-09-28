#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=0-08:00:00
#SBATCH --job-name=a4s_ag_lbl
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/ag_labels.log
# Argument_Graph stage 1: Qwen2.5-32B labels every introduction sentence of the benchA4S pairs.
# The server is brought up in this allocation and the labelling run talks to it on localhost.
set -e
cd ${SCISLOP_ROOT}
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
for i in $(seq 1 120); do
  curl -s -m 3 http://localhost:8000/v1/models >/dev/null && break
  sleep 15
done
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage labels --runs 3"
kill $SERVER 2>/dev/null || true
