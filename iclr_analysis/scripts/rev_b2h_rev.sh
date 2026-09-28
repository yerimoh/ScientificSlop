#!/bin/bash
# CycleReviewer (B2h) on the ICLR reviewer subset. 2 x 48GB GPUs. Mirrors bench165/scripts/run165_b2h.sh.
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
nvidia-smi -L
export CYCLEREV_ENDPOINT=http://127.0.0.1:8779/v1
$PY -m vllm.entrypoints.openai.api_server --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port 8779 --dtype half --gpu-memory-utilization 0.90 \
    --max-model-len 32768 > $R/logs/cyclerev_serve_rev.log 2>&1 &
CPID=$!
for i in $(seq 1 120); do curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $CYCLEREV_ENDPOINT/models >/dev/null 2>&1 || { echo "cyclereviewer never came up"; kill $CPID; exit 1; }
python3 $R/scripts/run_reviews_iclr.py --system b2h --reverse
kill $CPID; echo B2H_ICLR_REV_DONE
