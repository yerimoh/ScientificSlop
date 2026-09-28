#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --exclude=hockney
#SBATCH --time=7-00:00:00
#SBATCH --job-name=serve_cyclerev
#SBATCH --output=${SCISLOP_ROOT}/artifact-ai2science/Evaluation/_runlogs/cyclerev_server_sbatch.log
bash ${SCISLOP_ROOT}/artifact-ai2science/Evaluation/02_baselines_B/B2h_cyclereviewer/serve_cyclerev.sh
