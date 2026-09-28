#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=q-low
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=0-08:00:00
#SBATCH --job-name=a4s_figgate
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/figexp_gate.log
# fig_exposition step 2: the vision model decides which of the 1,448 cropped candidates is a method
# figure, the decision pairs165 made by hand. The same question is put to both corpora.
set -e
export HF_HUB_OFFLINE=1
export PYTHONPATH=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/scripts:$PYTHONPATH
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/figexp_gate.py
python3 scripts/run_figexp_a4s.py --transcribe
python3 scripts/run_figexp_a4s.py
