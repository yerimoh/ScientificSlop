#!/usr/bin/env bash
# Resume the interrupted 0915 pipeline after the existing reviewer jobs finish.
# This deliberately performs only the missing finalization steps.
set -euo pipefail

R=${SCISLOP_ROOT}/paper/draft_v6/review
LOG="$R/logs/pipeline_0915_resume.log"
JOB_IDS=${1:?usage: finalize_after_topup.sh JOB_ID[,JOB_ID...]}
cd "$R"
exec >>"$LOG" 2>&1

step() { echo; echo "===== [$(date '+%m-%d %H:%M')] $1"; }

step "resume: waiting for reviewer job(s) $JOB_IDS"
while squeue -h -j "$JOB_IDS" 2>/dev/null | grep -q .; do
  sleep 300
done

step "10 final figures + report"
for v in "" "--ours-agg3" "--bands" "--ours-agg3 --bands"; do
  python3 scripts/fig_rating_systems.py $v | tail -1
done
python3 scripts/fig_stairs_systems.py | tail -1
python3 scripts/report.py

D="${SCISLOP_ROOT}/paper/draft_v6/paper/real/_ICLR_2027__Scientific_Mold (2)/figures/main_figures"
cp results/fig_review_stairs.pdf "$D/fig3_review_stairs.pdf"
cp results/fig_rating_systems_agg3_bands.pdf "$D/fig3_review_vs_systems.pdf"
cp results/fig_stairs_systems.pdf "$D/fig3_review_stairs_systems.pdf"

step "PIPELINE_DONE"
