#!/bin/bash
# B2h one-shot bootstrap: weights -> server -> smoke -> full 165xR5 batch.
#
# The ONLY manual prerequisite (model is HF-gated, 'auto' approval):
#   1. visit https://huggingface.co/WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B
#      while logged in and click "Agree and access repository" (instant).
#   2. export HF_TOKEN=hf_...   (read token from hf.co/settings/tokens)
# Then:  bash bootstrap_b2h.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: export HF_TOKEN=hf_... first (and accept the license on the model page once)."
  exit 1
fi

echo "== [1/5] downloading WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B (~16GB) =="
python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download('WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B', token=os.environ['HF_TOKEN'])
print('weights at', p)
EOF

echo "== [2/5] submitting vLLM server (1x48GB, q-mid) =="
rm -f cyclerev_endpoint.txt
JOB=$(sbatch --parsable serve_cyclerev.sbatch)
echo "server job $JOB"

echo "== [3/5] waiting for endpoint health =="
for i in $(seq 1 120); do
  if [ -f cyclerev_endpoint.txt ] && curl -s --max-time 5 "$(cat cyclerev_endpoint.txt)/models" 2>/dev/null | grep -q cyclereviewer; then
    echo "server up: $(cat cyclerev_endpoint.txt)"; break
  fi
  sleep 20
  [ "$i" = 120 ] && { echo "server did not come up in 40min"; exit 1; }
done

echo "== [4/5] FA0001 reviewer smoke =="
python3 - <<'EOF'
import sys, json
ES = ('${SCISLOP_ROOT}/artifact-ai2science/'
      'Content_Mold/Evaluation_Surface_Area/results/code')
sys.path.insert(0, ES); sys.path.insert(0, '.')
import eslib, iter_run as ir
import cyclerev_reviewer as cr
rev, info = cr.perform_review(eslib.paper_tex(ir.paper_dir('FA0001')))
assert rev is not None, f"smoke FAILED: {info}"
print('smoke OK', info, '| Overall', rev['Overall'], rev['Decision'],
      '| reviewers', len(rev['Reviews']))
print(json.dumps(rev, indent=2)[:1200])
EOF

echo "== [5/5] launching full batch (8 shards, 165 papers, R5) =="
mkdir -p _runlogs
setsid bash ../run_b2h.sh > _runlogs/b2h_batch_master.log 2>&1 &
echo "batch running detached; logs: $HERE/_runlogs/b2h.s*.log"
echo "progress:  cat $HERE/progress_b2h.shard*.csv | grep -vc ^code"
