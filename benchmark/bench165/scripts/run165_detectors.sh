#!/bin/bash
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
DET=$ROOT/paper/draft_v6/claude/detectors
python3 $DET/binoculars.py --texts $B/scripts/texts.jsonl --out $B/results/binoculars.jsonl 2>&1 | tail -3
python3 $DET/detectgpt.py  --texts $B/scripts/texts.jsonl --out $B/results/detectgpt.jsonl  2>&1 | tail -3
echo DET165_DONE
