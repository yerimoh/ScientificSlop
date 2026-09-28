#!/bin/bash
# usage: run_detectors_2026.sh <shard> <nshard>  (1 GPU each) -> results/detectors/{binoculars,nts,detectgpt}.p26s<shard>.jsonl
set -x
ROOT=${SCISLOP_ROOT}; R=$ROOT/paper/draft_v6/review; DET=$ROOT/paper/draft_v6/claude/detectors
T=$R/records/texts_iclr2026.jsonl; S=$1; N=$2; O=$R/results/detectors
nvidia-smi -L
python3 $DET/binoculars.py --texts $T --out $O/binoculars.p26s$S.jsonl --shard $S --nshard $N
python3 $DET/detectgpt.py  --texts $T --out $O/detectgpt.p26s$S.jsonl  --shard $S --nshard $N
echo "DONE_P26_$S"
