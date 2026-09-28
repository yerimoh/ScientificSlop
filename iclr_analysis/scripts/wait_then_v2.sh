#!/bin/bash
# wait for the first pipeline to finish, then run pipeline_v2 (adds the 8 score-9 papers collected at 22:05)
R=${SCISLOP_ROOT}/paper/draft_v6/review
until grep -q PIPELINE_DONE $R/logs/pipeline_0915.log 2>/dev/null; do sleep 300; done
bash $R/scripts/pipeline_v2.sh
