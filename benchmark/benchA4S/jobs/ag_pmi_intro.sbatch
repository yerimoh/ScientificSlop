#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=0-08:00:00
#SBATCH --job-name=a4s_pmi
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/ag_pmi_intro.log
# Argument_Graph stage 2 on the item's own unit, the introduction. The labels are already cached, so
# this needs a card and no server, which is why it does not belong in the four-card allocation.
set -e
export HF_HUB_OFFLINE=1
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/run_slopA4S.py --checker argument_graph --extra "--stage pmi"
echo AG_PMI_INTRO_DONE
