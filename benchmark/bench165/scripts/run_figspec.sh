#!/bin/bash
set -x
B=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/bench165
until [ -f $B/results/slop/fig_specimen/manifest.jsonl ]; do sleep 60; done
python3 $B/scripts/figspec_vlm.py
touch $B/results/FIGSPEC_DONE
echo FIGSPEC_DONE
