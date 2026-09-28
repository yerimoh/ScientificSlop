#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=q-low
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=0-08:00:00
#SBATCH --job-name=a4s_figcensus
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/fig_census.log
# Seven candidate devices, asked of both benches with the same question set and the same reader.
set -e
export HF_HUB_OFFLINE=1
export PYTHONPATH=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/scripts:$PYTHONPATH
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/fig_census.py --bench benchA4S
python3 scripts/fig_census.py --bench bench165
echo FIG_CENSUS_DONE
