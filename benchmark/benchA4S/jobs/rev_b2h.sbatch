#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a4s_b2h
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/rev_b2h.log
# CycleReviewer over all 494 items. Its weights are its own reviewer model, so it needs its own
# server; the Qwen server the other jobs share cannot stand in for it.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CYCLEREV_ENDPOINT=http://127.0.0.1:8767/v1
$PY -m vllm.entrypoints.openai.api_server \
    --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port 8767 \
    --dtype half --gpu-memory-utilization 0.90 --max-model-len 32768 \
    > $B/jobs/cyclerev_serve.log 2>&1 &
CPID=$!
for i in $(seq 1 120); do curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 || { echo "cyclereviewer never came up"; kill $CPID; exit 1; }
export REVIEW_WORKERS=12
python3 $B/scripts/bench_reviewsA4S.py --system b2h
kill $CPID
echo B2H_A4S_DONE
