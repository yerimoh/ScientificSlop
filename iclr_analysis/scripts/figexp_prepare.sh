#!/bin/bash
# 0916. Picks (Qwen picker) for the captions on hand, then once the arXiv fetch has ended: captions for the new PDFs,
# picks for them, crops (8 shards), and the crops.done marker the VL job waits for. Log: logs/figexp_prepare.log
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
LOG=$R/logs/figexp_prepare.log; exec >> $LOG 2>&1
EP=$(cat ${SCISLOP_ROOT}/artifact-ai2science/_llm/llm_endpoint_qwen_pc.txt)
echo "[$(date '+%m-%d %H:%M')] picks round 1"; LLM_ENDPOINT=$EP python3 scripts/figexp_pick.py iclr
until grep -q "have_local" logs/figexp_fetch.log; do echo "[$(date '+%m-%d %H:%M')] waiting for pdf fetch ($(ls results/figexp/pdfs | wc -l))"; sleep 120; done
echo "[$(date '+%m-%d %H:%M')] pdf fetch retry (arXiv rate limits made the first pass drop some)"; python3 scripts/figexp_fetch_pdfs.py | tail -1; python3 scripts/figexp_fetch_pdfs.py | tail -1
echo "[$(date '+%m-%d %H:%M')] captions for new pdfs"; for s in 0 1 2 3 4 5 6 7; do python3 scripts/figexp_captions.py $s 8 & done; wait
echo "[$(date '+%m-%d %H:%M')] picks round 2"; LLM_ENDPOINT=$EP python3 scripts/figexp_pick.py iclr
echo "[$(date '+%m-%d %H:%M')] crops"; for s in 0 1 2 3 4 5 6 7; do python3 scripts/figexp_crop.py $s 8 & done; wait
python3 - <<'PY'
import json, glob, collections
rows = []
for f in sorted(glob.glob("results/figexp/manifest.s*.json")): rows += json.load(open(f))
print("manifest", len(rows), collections.Counter(r["status"].split(":")[0] for r in rows))
PY
touch results/figexp/crops.done; echo "[$(date '+%m-%d %H:%M')] PREPARE_DONE"
