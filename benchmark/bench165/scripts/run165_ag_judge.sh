#!/bin/bash
# One judge-ablation cell for Argument_Graph on bench165 (A3/E1).
# Usage: run165_ag_judge.sh <MODEL_TAG> <HF_MODEL> <SERVED_NAME> <TP> <PORT> [MAXLEN]
# Serves the judge with serve_alt.sh (own endpoint file, never touches llm_endpoint.txt),
# runs the labels stage with the tagged cache namespace, then the PMI stage (default scorer),
# writing to results/slop/argument_graph_<TAG>/.
set -x
TAG=$1; MODEL=$2; NAME=$3; TP=$4; PORT=$5; ML=${6:-32768}
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
LLM=$ROOT/artifact-ai2science/_llm
MODEL="$MODEL" NAME="$NAME" TP="$TP" PORT="$PORT" MAXLEN="$ML" bash $LLM/serve_alt.sh \
  > $B/results/supervisor/ag_judge_${TAG}_server.log 2>&1 &
SRV=$!
EP="http://127.0.0.1:${PORT}/v1"
for i in $(seq 1 120); do sleep 15; curl -s --max-time 5 "$EP/models" >/dev/null 2>&1 && break; done
curl -s --max-time 5 "$EP/models" || { echo SERVER_NEVER_HEALTHY_$TAG; kill $SRV; exit 1; }
export MODEL_TAG=$TAG LLM_ENDPOINT=$EP LLM_MODEL=$NAME
python3 $B/scripts/run_slop165.py --checker argument_graph --outsuffix _$TAG --extra "--stage labels --runs 3"
kill $SRV; sleep 20; pkill -f vllm.entrypoints 2>/dev/null; sleep 10
unset LLM_ENDPOINT
python3 $B/scripts/run_slop165.py --checker argument_graph --outsuffix _$TAG --extra "--stage pmi"
touch $B/results/AG_JUDGE_${TAG}_DONE
echo AG_JUDGE_${TAG}_DONE
