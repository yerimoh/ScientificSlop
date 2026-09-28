#!/bin/bash
# Fallback: serve Qwen 32B briefly and run the citation checker (used only if the login-node
# citation run dies after the main qwen job ends).
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
PY=${VLLM_PYTHON:-python3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export LLM_ENDPOINT=http://127.0.0.1:8765/v1
$PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --served-model-name qwen \
  --tensor-parallel-size 4 --port 8765 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  > $B/results/qwen_serve_cit.log 2>&1 &
QPID=$!
for i in $(seq 1 120); do curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
python3 $B/scripts/run_slop165.py --checker citation 2>&1 | tail -4
kill $QPID
echo CIT165_DONE
