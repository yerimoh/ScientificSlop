#!/bin/bash
# AI-Scientist reviewer (B3a) on the ICLR reviewer subset, Qwen2.5-32B-Instruct TP=4 on 48GB GPUs. Mirrors run165_qwen.sh.
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review
PY=${VLLM_PYTHON:-python3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export LLM_ENDPOINT=http://127.0.0.1:8778/v1
nvidia-smi -L
$PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --served-model-name qwen \
    --tensor-parallel-size 4 --port 8778 --dtype half --gpu-memory-utilization 0.92 --max-model-len 16384 > $R/logs/qwen_serve.log 2>&1 &
QPID=$!
for i in $(seq 1 120); do curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 || { echo "qwen never came up"; kill $QPID; exit 1; }
python3 $R/scripts/run_reviews_iclr.py --system b3a
kill $QPID; echo B3A_ICLR_DONE
