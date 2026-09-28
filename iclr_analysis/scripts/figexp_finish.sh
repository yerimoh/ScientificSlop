#!/bin/bash
# 0916. When every ok crop of the manifest has a transcript (any transcripts*.jsonl), score the item, rerun analysis,
# report and the six-item stairs figure. Log: logs/figexp_finish.log
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
LOG=$R/logs/figexp_finish.log; exec >> $LOG 2>&1
check() { python3 - <<'PY'
import json, glob, sys
R="${SCISLOP_ROOT}/paper/draft_v6/review"
need = {m["key"] for f in glob.glob(f"{R}/results/figexp/manifest.s*.json") for m in json.load(open(f)) if m.get("status") == "ok"}
done = {json.loads(l)["key"] for f in glob.glob(f"{R}/results/figexp/transcripts*.jsonl") for l in open(f)}
print(f"transcripts {len(need & done)}/{len(need)}"); sys.exit(0 if need <= done else 1)
PY
}
until check; do echo "[$(date '+%m-%d %H:%M')] waiting"; sleep 600; done
echo "[$(date '+%m-%d %H:%M')] transcripts complete, scoring"
python3 scripts/figexp_score.py 2>&1 | tail -5
python3 scripts/analyze.py 2>&1 | tail -30
python3 scripts/fig_stairs.py all6; python3 scripts/report.py 2>&1 | tail -2
echo "[$(date '+%m-%d %H:%M')] FIGEXP_FINISH_DONE"
