#!/bin/bash
# bench165 GPU job: detectors on 286 docs, then Qwen 32B (b3a + b3i human-side, citation slop),
# then CycleReviewer 8B (b2h human-side).
set -x
ROOT=${SCISLOP_ROOT}
SB=$ROOT/paper/draft_v6/scislopbench
B=$SB/bench165
DET=$ROOT/paper/draft_v6/claude/detectors
PY=${VLLM_PYTHON:-python3}
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false

wait_up () { for i in $(seq 1 120); do curl -s "$1/models" >/dev/null 2>&1 && return 0; sleep 15; done; return 1; }

# ---- phase 0: detectors -------------------------------------------------------
python3 $DET/binoculars.py --texts $B/scripts/texts.jsonl --out $B/results/binoculars.jsonl 2>&1 | tail -3
python3 $DET/detectgpt.py  --texts $B/scripts/texts.jsonl --out $B/results/detectgpt.jsonl  2>&1 | tail -3

# ---- phase 1: Qwen 32B --------------------------------------------------------
export LLM_ENDPOINT=http://127.0.0.1:8765/v1
$PY -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct --served-model-name qwen \
    --tensor-parallel-size 4 --port 8765 --gpu-memory-utilization 0.90 \
    --max-model-len 32768 > $B/results/qwen_serve.log 2>&1 &
QPID=$!
wait_up $LLM_ENDPOINT || { echo "qwen never came up"; kill $QPID; exit 1; }
python3 $B/scripts/bench_reviews.py --system b3i
python3 $B/scripts/bench_reviews.py --system b3a
python3 $B/scripts/run_slop165.py --checker citation 2>&1 | tail -5
kill $QPID; sleep 30

# ---- phase 2: CycleReviewer 8B ------------------------------------------------
export CYCLEREV_ENDPOINT=http://127.0.0.1:8766/v1 HF_HUB_OFFLINE=1
$PY -m vllm.entrypoints.openai.api_server \
    --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 1 --port 8766 --gpu-memory-utilization 0.85 \
    --max-model-len 50000 > $B/results/cyclerev_serve.log 2>&1 &
CPID=$!
wait_up $CYCLEREV_ENDPOINT || { echo "cyclereviewer never came up"; kill $CPID; exit 1; }
python3 $B/scripts/bench_reviews.py --system b2h
kill $CPID
echo BENCH165_DONE
