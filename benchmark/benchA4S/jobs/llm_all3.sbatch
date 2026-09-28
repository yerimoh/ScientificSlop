#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=a4s_llm3
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/llm_all3.log
# The previous run found the port held by a server that was still dying, read the stale endpoint
# file, and ran every item against nothing. The server now gets a free port of its own, the job
# writes that endpoint itself, and the job stops rather than running when the model does not answer.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
export PORT
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
export AG_LABEL_WORKERS=32 REVIEW_WORKERS=10 REVIEW_RAW=1
rm -rf results/reviews/b3a results/reviews/b3i results/reviews/b2h
python3 scripts/run_slopA4S.py --checker citation
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage labels --runs 3"
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage labels --runs 3 --unit intro_concl"
python3 scripts/bench_reviewsA4S.py --system b3a
python3 scripts/bench_reviewsA4S.py --system b3i
kill $SERVER 2>/dev/null || true
sleep 90
export HF_HUB_OFFLINE=1
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage pmi"
python3 scripts/run_slopA4S.py --checker argument_graph --outsuffix _wide \
  --extra "--stage pmi --unit intro_concl"
echo LLM_ALL3_DONE
