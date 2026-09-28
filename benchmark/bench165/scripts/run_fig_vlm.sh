#!/bin/bash
# E3 cell: fig_graph vision with a cross VLM. Usage: run_fig_vlm.sh <VLM_TAG> <MODEL_PATH>
set -x
TAG=$1; MODEL=$2
ROOT=${SCISLOP_ROOT}
FG=$ROOT/paper/draft_v6/slop/Artifacts/fig_graph
B=$ROOT/paper/draft_v6/scislopbench/bench165
export VLM_TAG=$TAG VLM_MODEL=$MODEL
cd $FG/code
mkdir -p $FG/results_$TAG
python3 measure.py --out $FG/results_$TAG
touch $B/results/FIGVLM_${TAG}_DONE
echo FIGVLM_${TAG}_DONE
