#!/bin/bash
#SBATCH --partition=vram48
#SBATCH --qos=${SLURM_QOS}
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=0-08:00:00
#SBATCH --job-name=a4s_figexp
#SBATCH --output=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/jobs/figexp_vlm.log
# fig_exposition on benchA4S. Qwen2.5-VL-32B first decides which captioned figure of each paper is
# the method figure, the decision pairs165 made by hand, then transcribes the ones it kept and the
# item is scored from those transcripts.
set -x
export HF_HUB_OFFLINE=1
export PYTHONPATH=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S/scripts:$PYTHONPATH
cd ${SCISLOP_ROOT}/paper/draft_v6/scislopbench/benchA4S
python3 scripts/figexp_gate.py
python3 scripts/run_figexp_a4s.py --transcribe
python3 scripts/run_figexp_a4s.py
