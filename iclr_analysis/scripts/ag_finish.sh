#!/bin/bash
# 0916. When the Argument_Graph caches cover the ICLR corpus, run the checker (PMI stage from cache, CPU) into
# results/slop/argument_graph/, then analyze + report. Log: logs/ag_finish.log
R=${SCISLOP_ROOT}/paper/draft_v6/review; D6=$(dirname $R)
LOG=$R/logs/ag_finish.log; exec >> $LOG 2>&1
cd $D6
until python3 slop/Argument/Argument_Graph/code/cache_status.py review/results/ag/papers_iclr.json; do echo "[$(date '+%m-%d %H:%M')] waiting for caches"; sleep 600; done
echo "[$(date '+%m-%d %H:%M')] caches complete, measuring"
cd $R
python3 scripts/run_slop_iclr.py --checker argument_graph --extra "--stage pmi" 2>&1 | tail -3
python3 scripts/analyze.py 2>&1 | tail -40
python3 scripts/report.py 2>&1 | tail -2
echo "[$(date '+%m-%d %H:%M')] AG_FINISH_DONE"
