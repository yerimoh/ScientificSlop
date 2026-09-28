#!/bin/bash
# Argument_Graph v3 over bench165 in ONE job: 32B label server (TP=4) -> labels -> server down -> PMI (1 GPU).
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
LLM=$ROOT/artifact-ai2science/_llm
EPF=$LLM/llm_endpoint.txt
mv -f $EPF $EPF.pre_ag 2>/dev/null
bash $LLM/serve_qwen.sh > $B/results/supervisor/ag_server.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  sleep 15
  [ -f $EPF ] && curl -s --max-time 5 "$(cat $EPF)/health" >/dev/null 2>&1 && break
done
curl -s --max-time 5 "$(cat $EPF)/health" || { echo SERVER_NEVER_HEALTHY; kill $SRV; exit 1; }
python3 $B/scripts/run_slop165.py --checker argument_graph --extra "--stage labels --runs 3"
kill $SRV; sleep 20; pkill -f vllm.entrypoints 2>/dev/null; sleep 10
python3 $B/scripts/run_slop165.py --checker argument_graph --extra "--stage pmi"
touch $B/results/AG165_DONE
echo AG165_DONE
