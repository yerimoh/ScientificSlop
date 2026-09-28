#!/bin/bash
# detect_rounds.sh ARM ROUND  -> results/detectors/<arm>/R<n>/{binoculars_faithful,fast_detectgpt,nts}.jsonl
# Same settings as bench165 (RESULTS_bench165_0911.md section 10b). Run on a GPU node:
#   sr 1 48 --qos=${SLURM_QOS} bash detect_rounds.sh a4_slop 1
set -x
ARM=$1; N=$2
ROOT=${SCISLOP_ROOT}
EOR=$ROOT/paper/draft_v6/Effects_of_revision
DET=$ROOT/paper/draft_v6/claude/detectors
T=$EOR/views/$ARM/R$N/texts.jsonl
O=$EOR/results/detectors/$ARM/R$N; mkdir -p $O
[ -s $O/binoculars_faithful.jsonl ] || python3 $DET/binoculars.py --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct --k 1 --window 512 --texts $T --out $O/binoculars_faithful.jsonl
[ -s $O/fast_detectgpt.jsonl ] || python3 $DET/fast_detectgpt.py --texts $T --out $O/fast_detectgpt.jsonl
[ -s $O/nts.jsonl ] || python3 $DET/nts.py --texts $T --out $O/nts.jsonl
echo DETECT_${ARM}_R${N}_DONE
