#!/bin/bash
# Snapshot every paper's score to CSV every five minutes, for as long as any review job is running, and for an
# hour after the last one. Survives the session (setsid). Log: logs/export_loop.log
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
idle=0
while [ $idle -lt 12 ]; do
  python3 scripts/export_scores.py >> logs/export_loop.log 2>&1 || echo "$(date '+%H:%M') export failed" >> logs/export_loop.log
  python3 scripts/normalize_scores.py >> logs/export_loop.log 2>&1 || echo "$(date '+%H:%M') normalise failed" >> logs/export_loop.log
  python3 scripts/make_workbook.py >> logs/export_loop.log 2>&1 || true
  if squeue -u $USER -h -o "%j" 2>/dev/null | grep -qE "r26|p26"; then idle=0; else idle=$((idle+1)); fi
  sleep 300
done
python3 scripts/export_scores.py >> logs/export_loop.log 2>&1
python3 scripts/normalize_scores.py >> logs/export_loop.log 2>&1
python3 scripts/make_workbook.py >> logs/export_loop.log 2>&1 || true
echo "EXPORT_LOOP_DONE $(date '+%m-%d %H:%M')" >> logs/export_loop.log
