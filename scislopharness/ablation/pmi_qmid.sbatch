#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pmi_daemon_abl
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/slopharness/ablation_components/_daemons/pmi_daemon.log
# Same daemon as ver1/api/final_1/_daemons/pmi.sbatch (Argument_Graph PMI stage, warm 7B LM on one GPU), on the q-mid QOS
# because q-high is at its billing cap (0922). AG_PMI_DAEMON=0 inside so the daemon computes instead of asking itself.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false AG_PMI_DAEMON=0
rm -f ${SCISLOP_ROOT}/paper/draft_v6/slop/_common/_pmi_cache/_requests/READY
exec python3 ${SCISLOP_ROOT}/paper/draft_v6/slopharness/ver1/api/code/pmi_daemon.py
