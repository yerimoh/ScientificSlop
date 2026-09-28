#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=0-06:00:00
#SBATCH --job-name=a4s_ag_pmi
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/ag_pmi.log
# Argument_Graph stage 2: pairwise PMI with the local 7B scorer, from the cached labels.
set -e
export HF_HUB_OFFLINE=1
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage pmi"
