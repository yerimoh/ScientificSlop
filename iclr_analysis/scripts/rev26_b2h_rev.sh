#!/bin/bash
# CycleReviewer / AI Scientist on the ICLR 2026 reviewer subset, rev worker. 2 GPUs.
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CYCLEREV_ENDPOINT=http://127.0.0.1:8801/v1
nvidia-smi -L
$PY -m vllm.entrypoints.openai.api_server --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port 8801 --dtype half --gpu-memory-utilization 0.90 \
    --max-model-len 16384 > $R/logs/serve26_b2h_rev.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 || { echo "server never came up"; kill $SPID; exit 1; }
python3 $R/scripts/run_reviews_iclr.py --system b2h --subset reviewer_subset_2026.json --views views2026 --outdir b2h_2026 --reverse
kill $SPID; echo "DONE26_b2h_rev"
