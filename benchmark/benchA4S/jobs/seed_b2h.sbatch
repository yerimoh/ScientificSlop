#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --time=0-08:00:00
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/seed_b2h_%j.log
# CycleReviewer, one more draw. It generates at temperature 0.4, so its rating is a sample.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
export CYCLEREV_ENDPOINT=http://127.0.0.1:${PORT}/v1
$PY -m vllm.entrypoints.openai.api_server \
    --model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B --served-model-name cyclereviewer \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port ${PORT} \
    --dtype half --gpu-memory-utilization 0.90 --max-model-len 32768 > $B/jobs/cyc_${RUN}.log 2>&1 &
CPID=$!
for i in $(seq 1 160); do curl -s -m 3 $CYCLEREV_ENDPOINT/models | grep -q '"id"' && break; sleep 15; done
curl -s -m 3 $CYCLEREV_ENDPOINT/models | grep -q '"id"' || { echo ABORT; kill $CPID; exit 1; }
cd $B
export REVIEW_WORKERS=12
python3 scripts/bench_reviewsA4S.py --system b2h --run ${RUN}
kill $CPID
echo B2H_RUN_${RUN}_DONE
