#!/bin/bash
# usage: rev26_worker.sh <b2h|b3a> <shard> <nshard> <port>
# One vLLM server per worker with its own outlines cache. Four servers sharing ~/.cache/outlines corrupted its
# sqlite file on 0917 and every request then returned 500, so the cache directory is per job from here on.
set -x
SYS=$1; SHARD=$2; NSHARD=$3; PORT=$4; REV=${5:-}; SUBSET=${6:-reviewer_subset_2026.json}
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OUTLINES_CACHE_DIR=$R/logs/_outlines_${SYS}_${SHARD}${REV}_$(echo $SUBSET | md5sum | cut -c1-6)
rm -rf $OUTLINES_CACHE_DIR; mkdir -p $OUTLINES_CACHE_DIR
nvidia-smi -L
if [ "$SYS" = b2h ]; then
  MODEL=WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B; SERVED=cyclereviewer; TP=1; export CYCLEREV_ENDPOINT=http://127.0.0.1:$PORT/v1; EP=$CYCLEREV_ENDPOINT
else
  MODEL=Qwen/Qwen2.5-32B-Instruct; SERVED=qwen; TP=2; export LLM_ENDPOINT=http://127.0.0.1:$PORT/v1; EP=$LLM_ENDPOINT
fi
$PY -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name $SERVED \
    --tensor-parallel-size $TP --disable-custom-all-reduce --enforce-eager --port $PORT --dtype half \
    --gpu-memory-utilization 0.90 --max-model-len 16384 > $R/logs/serve26_${SYS}_${SHARD}.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -s $EP/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $EP/models >/dev/null 2>&1 || { echo "server never came up"; kill $SPID; exit 1; }
# one real request must succeed before the whole subset is walked
python3 $R/scripts/run_reviews_iclr.py --system $SYS --subset $SUBSET --views views2026 \
        --outdir ${SYS}_2026 --shard $SHARD --nshard $NSHARD $REV
kill $SPID; echo "DONE26_${SYS}_${SHARD}"
