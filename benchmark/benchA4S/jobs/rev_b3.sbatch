#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=q-low
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a4s_b3a
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/rev_b3.log
# AI-Scientist, pulled out of the long job so it does not wait behind seven hours of sentence
# labelling. The CMU reviewer was dropped on request. Two cards is enough for the 32B at tensor-parallel 2, and the port
# is picked free so this server cannot be confused with the one the other allocation is running.
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
echo "[smoke] $ANS"
echo "$ANS" | grep -qi "ok" || { echo "SERVER NOT ANSWERING, ABORT"; kill $SERVER; exit 1; }
cd $B
export REVIEW_WORKERS=10 REVIEW_RAW=1
python3 scripts/bench_reviewsA4S.py --system b3a
kill $SERVER 2>/dev/null || true
echo B3_DONE
