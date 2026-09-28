#!/bin/bash
# 0916 open-model pilot: Qwen2.5-32B-Instruct (vLLM, _llm/serve_qwen_edit.sbatch) as the text-in/text-out editor of arm
# a1_base, one round, all 143 papers, 4 shards in parallel. Storage suffix .qwen32b (runs/, progress/, results/slop/).
#   bash run_open_pilot.sh [nshard]
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); N=${1:-4}; ARM=${2:-a1_base}
export EOR_BACKEND=vllm EOR_MODEL=Qwen2.5-32B-Instruct EOR_TAG=.qwen32b
for i in $(seq 0 $((N-1))); do
  setsid nohup python3 $HERE/run_arm.py --arm $ARM --rounds 1 --shard $i --nshard $N > $ROOT/logs/$ARM.qwen32b.shard$i.log 2>&1 &
done
echo "launched $N shards"
