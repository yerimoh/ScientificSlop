#!/bin/bash
# After the main 2026 reviewer jobs end, start one server and retry only the papers whose AI Scientist review did
# not parse, at a raised generation limit. Also covers the multi-year corpus and the benchmark side.
set -x
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
while squeue -u $USER -h -o "%j" 2>/dev/null | grep -q r26; do sleep 300; done
PY=${VLLM_PYTHON:-python3}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OUTLINES_CACHE_DIR=$R/logs/_outlines_retry; rm -rf $OUTLINES_CACHE_DIR; mkdir -p $OUTLINES_CACHE_DIR
export LLM_ENDPOINT=http://127.0.0.1:8850/v1
$PY -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --served-model-name qwen \
    --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager --port 8850 --dtype half \
    --gpu-memory-utilization 0.90 --max-model-len 16384 > $R/logs/serve_retry_b3a.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 && break; sleep 15; done
curl -s $LLM_ENDPOINT/models >/dev/null 2>&1 || { echo "server never came up"; kill $SPID; exit 1; }
python3 scripts/retry_b3a_failed.py --outdir b3a_2026 --views views2026 --limit 6000
python3 scripts/retry_b3a_failed.py --outdir b3a      --views views     --limit 6000
kill $SPID
python3 scripts/export_scores.py; python3 scripts/normalize_scores.py; python3 scripts/make_workbook.py
echo "RETRY_B3A_DONE"
