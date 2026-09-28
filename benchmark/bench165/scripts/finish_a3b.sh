#!/bin/bash
# Supervisor for the A3 leftovers: E4 citation-claims judges + E3 fig_graph VLM cross.
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
FG=$ROOT/paper/draft_v6/slop/Artifacts/fig_graph
SUP=$B/results/supervisor
need="CIT_CLAIMS_qwen32b_DONE CIT_CLAIMS_mistral24b_DONE CIT_CLAIMS_gemma27b_DONE"   # fig_graph VLM cross (E3) shelved on user instruction 0912
for i in $(seq 1 288); do
  miss=0; for m in $need; do [ -f $B/results/$m ] || miss=$((miss+1)); done
  echo "$(date +%m%d-%H:%M) a3b missing=$miss" >> $SUP/a3b_status.log
  [ "$miss" -eq 0 ] && break
  if [ "$(squeue -u $USER -h -o %j | grep -cE '^(cit_|figvlm_)')" -eq 0 ] && [ "$i" -gt 3 ]; then break; fi
  sleep 300
done
for t in qwen32b mistral24b gemma27b; do
  [ -f $B/results/slop/citation_claims_$t/papers.jsonl ] && \
    python3 $B/scripts/summarize_pairs.py $B/results/slop/citation_claims_$t/papers.jsonl \
      $B/results/slop/cit_claims_${t}_summary.json claims_vague_score >> $SUP/a3b_status.log 2>&1
done
python3 - <<'PY' >> $SUP/a3b_status.log 2>&1
import json, glob, os, statistics as st
FG = "${SCISLOP_ROOT}/paper/draft_v6/slop/Artifacts/fig_graph"
out = {}
for d, tag in [(f"{FG}/results", "qwenvl32b"), (f"{FG}/results_mistralvl24b", "mistralvl24b"), (f"{FG}/results_gemma3vl27b", "gemma3vl27b")]:
    p = f"{d}/papers.jsonl"
    if not os.path.exists(p): continue
    rows = [json.loads(l) for l in open(p)]
    r = {}
    for c in ("AI", "HU"):
        v = [x["undescribed_box_rate"] for x in rows if x["corpus"] == c and isinstance(x.get("undescribed_box_rate"), (int, float))]
        r[c] = {"mean": round(st.mean(v), 3) if v else None, "n": len(v)}
    out[tag] = r
json.dump(out, open("${SCISLOP_ROOT}/paper/draft_v6/scislopbench/bench165/results/slop/fig_vlm_ablation_summary.json", "w"), indent=1)
print(json.dumps(out))
PY
touch $B/results/A3B_DONE
