#!/bin/bash
# usage: run_detectors_new.sh <texts.jsonl> <tag>   (1 GPU) -> results/detectors/{binoculars,detectgpt,nts}.<tag>.jsonl
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review; DET=$ROOT/paper/draft_v6/claude/detectors
T=$1; TAG=$2; O=$R/results/detectors
nvidia-smi -L
python3 $DET/binoculars.py --texts $T --out $O/binoculars.$TAG.jsonl
python3 $DET/nts.py        --texts $T --out $O/nts.$TAG.jsonl
python3 $DET/detectgpt.py  --texts $T --out $O/detectgpt.$TAG.jsonl
echo "DONE_NEW_$TAG"
