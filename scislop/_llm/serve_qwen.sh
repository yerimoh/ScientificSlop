#!/bin/bash
# Unified open-source LLM server for ALL mold MEASUREMENT / EXTRACTION / JUDGE calls.
# (Paper-REWRITE / file-editing stays on the Haiku CLI agent — it is the experimental subject.)
#
# Launch (4x48GB, tensor-parallel 4):
#   screen -dmS qwen_server bash -c 'sr 4 48 --qos=${SLURM_QOS} bash artifact-ai2science/_llm/serve_qwen.sh'
# Fallback if the 48GB queue is busy — 24GB shards need a smaller model:
#   MODEL=Qwen/Qwen2.5-14B-Instruct TP=2 ... sr 2 24 bash serve_qwen.sh
#
# Writes the reachable endpoint to _llm/llm_endpoint.txt so clients on the login node find it.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${VLLM_PYTHON:-python3}

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TP="${TP:-4}"                      # tensor-parallel = number of GPUs
PORT="${PORT:-8765}"
UTIL="${UTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"

# TP>1 under srun: fork-based workers hit "CUDA unknown error" -> must spawn.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

HOST_IP=$(hostname -i | awk '{print $1}')
echo "http://${HOST_IP}:${PORT}/v1" > "$HERE/llm_endpoint.txt"
echo "[serve] model=$MODEL tp=$TP endpoint=$(cat "$HERE/llm_endpoint.txt")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --host 0.0.0.0 --port "$PORT" \
  --guided-decoding-backend lm-format-enforcer \
  --disable-log-requests
# NOTE: the default guided-decoding backend ('outlines') imports pyairports, which is a broken
# 0.0.1 placeholder on this mirror -> every /chat/completions returned 500. lm-format-enforcer
# is installed and avoids that import entirely.
