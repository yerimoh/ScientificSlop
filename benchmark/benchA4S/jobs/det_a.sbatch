#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=q-low
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=0-10:00:00
#SBATCH --job-name=a4s_det_a
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/det_a.log
# The three cheap text detectors, in one allocation and in this order, because Binoculars and NTS
# both score with falcon-7b and the second run reads it from the page cache the first one warmed.
set -x
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/benchA4S
DET=$ROOT/paper/draft_v6/claude/detectors
T=$B/scripts/texts.jsonl
R=$B/results/baselines
python3 $DET/binoculars.py --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct \
  --k 1 --window 512 --texts $T --out $R/binoculars_faithful.jsonl
python3 $DET/nts.py --model tiiuae/falcon-7b --texts $T --out $R/nts.jsonl
python3 $DET/binoculars.py --texts $T --out $R/binoculars.jsonl
python3 $DET/fast_detectgpt.py --texts $T --out $R/fast_detectgpt.jsonl
echo DET_A_DONE
