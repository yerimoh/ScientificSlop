#!/bin/bash
# CycleReviewer on A6000 48GB with the exact bench5-proven settings (bf16, TP=1, 50k len).
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
PY=${VLLM_PYTHON:-python3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export CYCLEREV_ENDPOINT=http://127.0.0.1:8767/v1
$PY -m vllm.entrypoints.openai.api_server \
    --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 1 --port 8767 --gpu-memory-utilization 0.85 \
    --max-model-len 50000 > $B/results/cyclerev_serve_high.log 2>&1 &
CPID=$!
for i in $(seq 1 120); do curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 || { echo "cyclereviewer never came up"; kill $CPID; exit 1; }
python3 $B/scripts/bench_reviews.py --system b2h
kill $CPID
echo B2H165_DONE
