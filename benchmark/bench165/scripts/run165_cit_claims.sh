#!/bin/bash
# E4 cell: citation --claims observation layer under one judge.
# Usage: run165_cit_claims.sh <TAG> <HF_MODEL> <NAME> <TP> <PORT> [MAXLEN]
set -x
TAG=$1; MODEL=$2; NAME=$3; TP=$4; PORT=$5; ML=${6:-32768}
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
LLM=$ROOT/artifact-ai2science/_llm
MODEL="$MODEL" NAME="$NAME" TP="$TP" PORT="$PORT" MAXLEN="$ML" bash $LLM/serve_alt.sh \
  > $B/results/supervisor/cit_claims_${TAG}_server.log 2>&1 &
SRV=$!
EP="http://127.0.0.1:${PORT}/v1"
for i in $(seq 1 120); do sleep 15; curl -s --max-time 5 "$EP/models" >/dev/null 2>&1 && break; done
curl -s --max-time 5 "$EP/models" || { echo SERVER_NEVER_HEALTHY_$TAG; kill $SRV; exit 1; }
export MODEL_TAG=$TAG LLM_ENDPOINT=$EP LLM_MODEL=$NAME
python3 $B/scripts/run_slop165.py --checker citation --outsuffix _claims_$TAG --extra=--claims
kill $SRV; sleep 15; pkill -f vllm.entrypoints 2>/dev/null
touch $B/results/CIT_CLAIMS_${TAG}_DONE
echo CIT_CLAIMS_${TAG}_DONE
