#!/bin/bash
# Binoculars in the configuration Table 2 actually reports (bench165 results/binoculars.jsonl):
# Qwen2.5-7B / Qwen2.5-7B-Instruct, 512-token windows, k=6, the binoculars.py defaults.
# One GPU job, all arms and rounds, so the model pair loads once.
set -x
ROOT=${SCISLOP_ROOT}
EOR=$ROOT/paper/draft_v6/Effects_of_revision
DET=$ROOT/paper/draft_v6/claude/detectors
for ARM in a1_base a2_code a3_review a4_slop; do for N in 1 3; do
  T=$EOR/views/$ARM/R$N/texts.jsonl; O=$EOR/results/detectors/$ARM/R$N; mkdir -p $O
  [ -s $O/binoculars.jsonl ] || python3 $DET/binoculars.py --texts $T --out $O/binoculars.jsonl
done; done
echo BINO_DEFAULT_ALL_DONE
