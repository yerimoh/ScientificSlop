#!/bin/bash
# usage: run_detectors_iclr.sh <job: d1|d2|d3>   (each job expects 2 GPUs)
# d1/d2: Binoculars + DetectGPT shard 0/1 of 2 (same settings as bench165 Table 2 runs)
# d3   : Fast-DetectGPT then NTS (no sharding in those scripts)
set -x
ROOT=${SCISLOP_ROOT}
R=$ROOT/paper/draft_v6/review
DET=$ROOT/paper/draft_v6/claude/detectors
T=$R/records/texts_iclr.jsonl
O=$R/results/detectors
nvidia-smi -L
case "$1" in
  d1|d2) S=$([ "$1" = d1 ] && echo 0 || echo 1)
    python3 $DET/binoculars.py --texts $T --out $O/binoculars.s$S.jsonl --shard $S --nshard 2
    python3 $DET/detectgpt.py  --texts $T --out $O/detectgpt.s$S.jsonl  --shard $S --nshard 2 ;;
  d3)
    python3 $DET/fast_detectgpt.py --texts $T --out $O/fast_detectgpt.jsonl
    python3 $DET/nts.py            --texts $T --out $O/nts.jsonl ;;
  d4)  # NTS rerun on texts with >= 200 words (nts.py crashes on an empty body: iclr2017_BJAA4wKxg)
    python3 $DET/nts.py            --texts $R/records/texts_iclr_det.jsonl --out $O/nts.jsonl ;;
esac
echo "DONE_$1"
