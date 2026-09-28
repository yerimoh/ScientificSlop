#!/bin/bash
# 0916. When the Argument_Graph label and PMI caches cover every Haiku round tree, measure the item on every
# round (measure_rounds.py --items argument_graph, PMI stage from cache, CPU) and aggregate to
# results/summary/effects_ag.{json,md}, effects_common_ag.*, interactions_ag.*. Log: logs/ag_finish.log
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); D6=$(dirname "$ROOT")
LOG=$ROOT/logs/ag_finish.log; exec >> $LOG 2>&1
cd $D6
until python3 slop/Argument/Argument_Graph/code/cache_status.py Effects_of_revision/results/ag/papers_trees.json; do echo "[$(date '+%m-%d %H:%M')] waiting for caches"; sleep 600; done
echo "[$(date '+%m-%d %H:%M')] caches complete, measuring"
cd $HERE
for a in a1_base a2_code a3_review a4_slop; do python3 measure_rounds.py --arm $a --round 1 --round 2 --round 3 --items argument_graph; done
for it in macro_redund xsec_ref citation evidence_gap; do python3 measure_rounds.py --arm a4s_$it --round 1 --items argument_graph; done
EOR_ITEMS=argument_graph EOR_OUT=_ag python3 aggregate.py > $ROOT/logs/aggregate_ag.log 2>&1
echo "[$(date '+%m-%d %H:%M')] AG_FINISH_DONE"
