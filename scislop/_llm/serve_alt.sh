#!/bin/bash
# Second measurement server, for cross-model replication of an item.
# Writes its own endpoint file so the Qwen server on 8765 stays untouched.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${VLLM_PYTHON:-python3}
MODEL="${MODEL:?set MODEL}"
NAME="${NAME:?set NAME}"
TP="${TP:-4}"
PORT="${PORT:-8766}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1        # weights are already local; never let vLLM pull a consolidated.safetensors
HOST_IP=$(hostname -i | awk '{print $1}')
echo "http://${HOST_IP}:${PORT}/v1" > "$HERE/llm_endpoint_${NAME}.txt"
echo "[serve] model=$MODEL name=$NAME endpoint=$(cat "$HERE/llm_endpoint_${NAME}.txt")"
exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --host 0.0.0.0 --port "$PORT" \
  --guided-decoding-backend lm-format-enforcer \
  --disable-log-requests
