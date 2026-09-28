#!/usr/bin/env bash
# Create the working-tree layout the scripts expect under $SCISLOP_ROOT, pointing back into this repository.
#
# Every script in this release was written against one working tree and addresses its parts by relative path
# below a single root (paper/draft_v6/slop, paper/draft_v6/scislopbench/bench165, fars/papers, ...). Rather than
# rewriting several hundred path constants, the release keeps those relative paths and resolves the root from the
# environment variable SCISLOP_ROOT. This script builds that root out of symlinks into the repository, so the
# code runs unchanged:
#
#     export SCISLOP_ROOT=$HOME/scislop_root
#     bash scripts/make_root.sh
#     python3 scislopbench/build/materialize_bench.py      # benchmark views + items from the released Parquet tables
#     python3 benchmark/run_measure_release.py --half fars --checker xsec_ref
#
# Data that is not in this repository (the FARS dump, arXiv e-prints, the Agents4Science PDFs, model weights,
# measurement caches) has to be placed under the same root; see README.md, section "Data root layout".
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SCISLOP_ROOT:?set SCISLOP_ROOT to the directory that will hold the working-tree layout}"
link() {  # link <root-relative target> <repo-relative source>
  local t="$ROOT/$1" s="$REPO/$2"
  mkdir -p "$(dirname "$t")"
  if [ -e "$t" ] && [ ! -L "$t" ]; then echo "keep existing $t"; return; fi
  ln -sfn "$s" "$t"; echo "$1 -> $2"
}
link paper/draft_v6/slop                                             scislop
link artifact-ai2science/_llm                                        scislop/_llm
link paper/draft_v6/scislopbench/data/scripts                        benchmark/construction/common
link paper/draft_v6/scislopbench/data/pairs165_0911/scripts          benchmark/construction/pairs_fars
link paper/draft_v6/scislopbench/data/Agents4Science/scripts         benchmark/construction/pairs_a4s
link paper/draft_v6/scislopbench/bench165/scripts                    benchmark/bench165/scripts
link paper/draft_v6/scislopbench/benchA4S/scripts                    benchmark/benchA4S/scripts
link paper/draft_v6/scislopbench/benchA4S/jobs                       benchmark/benchA4S/jobs
link paper/draft_v6/claude/detectors                                 baselines/detectors
link artifact-ai2science/Evaluation/02_baselines_B/B2h_cyclereviewer baselines/reviewers/cyclereviewer
link artifact-ai2science/Evaluation/02_baselines_B/B3a_ai_scientist  baselines/reviewers/ai_scientist
link artifact-ai2science/Evaluation/02_baselines_B/B3i_cmu           baselines/reviewers/cmu_reviewer
link paper/draft_v6/review/scripts                                   iclr_analysis/scripts
link artifact-ai2science/Evaluation/ICLR                             iclr_analysis/collect/iclr_2017_2025
link artifact-ai2science/Evaluation/ICLR2026_Pangram                 iclr_analysis/collect/iclr_2026
link paper/draft_v6/slopharness/ver1/temp/code                       scislopharness/harness
link paper/draft_v6/slopharness/ver1/temp/skill                      scislopharness/skill
link paper/draft_v6/slopharness/ver1/api/code                        scislopharness/drivers
link paper/draft_v6/Effects_of_revision/code                         scislopharness/revision_baselines
link paper/draft_v6/slopharness/ablation_components/code             scislopharness/ablation/code
link paper/draft_v6/slopharness/ablation_components/skills           scislopharness/ablation/skills
mkdir -p "$ROOT/paper/draft_v6/scislopbench/bench165" "$ROOT/paper/draft_v6/scislopbench/benchA4S" "$ROOT/fars/papers"
echo "root ready at $ROOT"
