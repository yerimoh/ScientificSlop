#!/bin/bash
B=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/bench165
FG=${SCISLOP_ROOT}/paper/draft_v6/slop/Artifacts/fig_graph
{
echo
echo "### Automatic snapshot ($(date '+%m%d %H:%M %Z'))"
echo
echo '```'
echo "[queue]"; squeue -u $USER -o "%.9i %.13j %.8T %.10M" 2>/dev/null
echo "[fig vlm cache] mistralvl24b=$(ls $FG/roi_extract/vis_mistralvl24b_*.json 2>/dev/null | wc -l)/96  gemma3vl27b=$(ls $FG/roi_extract/vis_gemma3vl27b_*.json 2>/dev/null | wc -l)/96"
echo "[markers]"; ls $B/results | grep -E "FIGVLM|CIT_CLAIMS|A3B_DONE" || echo none
for t in qwen32b mistral24b gemma27b; do
  d=$B/results/slop/citation_claims_$t
  [ -f $d/papers.jsonl ] && echo "[cit_claims $t] papers=$(wc -l < $d/papers.jsonl)" || echo "[cit_claims $t] no output yet"
done
echo "[llm cache size] $(ls ${SCISLOP_ROOT}/paper/draft_v6/slop/_common/_llm_cache | wc -l) calls"
echo '```'
} >> $B/RESULTS_bench165_0911.md
