#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --time=0-10:00:00
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/seed_b3a_%j.log
# One shard of the AI-Scientist run. Its own server on its own free port, because the 32B at
# tensor-parallel 2 saturates two cards and a second shard on the same server would only queue.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
export PORT TP=2
cd $ROOT
bash artifact-ai2science/_llm/serve_qwen.sh &
SERVER=$!
EP="http://$(hostname -i | awk '{print $1}'):${PORT}/v1"
export LLM_ENDPOINT="$EP"
for i in $(seq 1 160); do
  curl -s -m 3 "${EP}/models" | grep -q '"id"' && break
  sleep 15
done
ANS=$(curl -s -m 60 "${EP}/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":5}')
echo "$ANS" | grep -qi "ok" || { echo "SERVER NOT ANSWERING, ABORT"; kill $SERVER; exit 1; }
cd $B
export REVIEW_WORKERS=16 REVIEW_RAW=1
python3 scripts/bench_reviewsA4S.py --system b3a --run ${RUN} --shard ${SHARD} --nshard ${NSHARD}
kill $SERVER 2>/dev/null || true
echo B3A_SHARD_${SHARD}_DONE
