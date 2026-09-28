#!/bin/bash
# One PMI-scorer ablation cell for Argument_Graph on bench165 (A3/E2).
# Usage: run165_ag_pmi.sh <SUFFIX> <PMI_MODEL>. Labels come from the qwen32b cache (default MODEL_TAG).
set -x
SUF=$1; PM=$2
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
export PMI_MODEL=$PM
python3 $B/scripts/run_slop165.py --checker argument_graph --outsuffix _pmi_$SUF --extra "--stage pmi"
touch $B/results/AG_PMI_${SUF}_DONE
echo AG_PMI_${SUF}_DONE
