#!/bin/bash
# CycleReviewer / AI Scientist on the ICLR 2026 reviewer subset, fwd worker. 2 GPUs.
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export LLM_ENDPOINT=http://127.0.0.1:8810/v1
nvidia-smi -L
$PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --served-model-name qwen \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port 8810 --dtype half --gpu-memory-utilization 0.90 \
    --max-model-len 16384 > $R/logs/serve26_b3a_fwd.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 || { echo "server never came up"; kill $SPID; exit 1; }
python3 $R/scripts/run_reviews_iclr.py --system b3a --subset reviewer_subset_2026.json --views views2026 --outdir b3a_2026 
kill $SPID; echo "DONE26_b3a_fwd"
