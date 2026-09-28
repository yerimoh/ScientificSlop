#!/bin/bash
# CycleReviewer-ML-Llama-3.1-8B server for the B2h baseline (review generation ONLY;
# judge stays on the shared Qwen server, rewrite stays on the Haiku CLI).
#
# Launch (1x48GB is plenty for 8B + 50k-token window):
#   sbatch serve_cyclerev.sbatch
# The model is HF-gated ('auto' approval): needs HF_TOKEN with the license
# accepted once at https://huggingface.co/WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B
#
# Writes the reachable endpoint to B2h_cyclereviewer/cyclerev_endpoint.txt.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${VLLM_PYTHON:-python3}

MODEL="${MODEL:-WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B}"
TP="${TP:-1}"
PORT="${PORT:-8766}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-50000}"          # canonical CycleReviewer max_model_len

export VLLM_WORKER_MULTIPROC_METHOD=spawn   # needed for TP>1 under srun; harmless at TP=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

HOST_IP=$(hostname -i | awk '{print $1}')
echo "http://${HOST_IP}:${PORT}/v1" > "$HERE/cyclerev_endpoint.txt"
echo "[serve] model=$MODEL tp=$TP endpoint=$(cat "$HERE/cyclerev_endpoint.txt")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name cyclereviewer \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --host 0.0.0.0 --port "$PORT" \
  --guided-decoding-backend lm-format-enforcer \
  --disable-log-requests
# lm-format-enforcer: the default 'outlines' backend imports pyairports, broken on
# this mirror (same fix as _llm/serve_qwen.sh).
