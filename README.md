# Science or Slop? — Benchmarking and Mitigating Scientific Slop in AI-Generated Papers

Code and data for the paper *Science or Slop?: Benchmarking and Mitigating Scientific Slop in AI-Generated Papers*
(under double-blind review, ICLR 2027). Everything in this repository is anonymized for review.

The paper makes three contributions, and the repository is organized around them:

| Contribution | Paper | Folder |
|---|---|---|
| **SciSlop**: six measures of scientific slop across Structure, Argument and Artifacts | §3.1, Table 1, Appendix A | [`scislop/`](scislop/) |
| **SciSlopBench**: 390 AI-generated papers paired with human-written papers | §3.2, Tables 2–3, Appendix B | [`scislopbench/`](scislopbench/) (data) and [`benchmark/`](benchmark/) (construction and evaluation code) |
| **SciSlopHarness**: a harness that guides a fixed LLM to revise slop where the experiment records support the change | §4, §5.3–5.4, Figs. 5–6, Appendices C–D | [`scislopharness/`](scislopharness/) |

Two more folders hold the comparison systems and the ICLR analysis:

| | Paper | Folder |
|---|---|---|
| Token-level detectors and automated reviewers used as baselines | §5.1, Table 2 | [`baselines/`](baselines/) |
| Slop against ICLR review scores and decisions, 2017–2026 | §5.2, Fig. 4, Appendix E | [`iclr_analysis/`](iclr_analysis/) |

## Quick start

The released benchmark is three Parquet tables; the measures, the tables of the paper and the deterministic measures
run from them without any external data.

```bash
pip install -r requirements.txt          # numpy scipy pandas pyarrow matplotlib ... (GPU packages optional)

# 1. Table 2 / Table 3 of the paper from the released scores (pair accuracy, AUROC)
python3 - <<'EOF'
import pandas as pd
s = pd.read_parquet("scislopbench/data/scores.parquet")
def pair_acc(col, direction=+1):
    w = s.pivot_table(index="pair_id", columns="label", values=col).dropna()
    d = (w[1] - w[0]) * direction
    return round(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d), 3), len(d)
print("SciSlop", pair_acc("scislop_aggregate"))        # (0.859, 390)
print("Binoculars", pair_acc("binoculars", -1))         # (0.687, 390)
EOF

# 2. Re-run the deterministic measures on the released paper bodies
export SCISLOP_ROOT=$HOME/scislop_root                 # any empty directory
bash scripts/make_root.sh                              # symlinks this repo into the layout the scripts expect
python3 scislopbench/build/materialize_bench.py        # writes bench165/ and benchA4S/ items + views under $SCISLOP_ROOT
python3 benchmark/run_measure_release.py --half fars --checker xsec_ref
python3 benchmark/run_measure_release.py --half a4s  --checker macro_redund
# -> $SCISLOP_ROOT/paper/draft_v6/scislopbench/{bench165,benchA4S}/results/slop/<checker>/papers.jsonl
```

`benchmark/benchA4S/scripts/table_pooled.py` then prints Tables 2 and 3 from those `papers.jsonl` files together
with the detector and reviewer outputs (the released `scores.parquet` was built from exactly these files by
`scislopbench/build/build_hf_release.py`).

## What reproduces from this repository alone

We re-ran the four deterministic measures on the released body views (`papers.parquet`) and compared every score
with `scores.parquet`:

| Measure | Identical scores (of 780) | Pair accuracy, re-run vs. released |
|---|---|---|
| Cross-section references | 759 | 0.906 vs. 0.905 |
| Macro redundancy | 767 | 0.723 vs. 0.723 |
| Citation isolation | 778 | 0.794 vs. 0.793 |
| Evidence gap | 701 | 0.640 vs. 0.764 |

The differences have one cause: the released view is the paper body without its appendix, while the benchmark run
read the full source. Evidence gap looks for concrete examples "anywhere in the paper, appendix included", so papers
whose only exhibits sit in the appendix score 1.0 on the body view and 0.0 in the paper; Cross-section references
additionally counts pointers into the appendix. The two model-based measures (Argument graph, Figure exposition) need
Qwen2.5-32B-Instruct, Qwen2.5-7B-Instruct and Qwen2.5-VL-32B-Instruct and the figure crops; their code and prompts
are included, their per-paper scores are in `scores.parquet`.

## Paper-to-code map

| Paper item | What it needs | Code |
|---|---|---|
| Table 1, App. A: measure definitions | – | `scislop/<plane>/<measure>/code/measure.py`, prompts `PROMPT_*.txt` |
| Table 2, Table 3: discrimination on 390 pairs | `scores.parquet` or the `papers.jsonl` runs | `benchmark/benchA4S/scripts/table_pooled.py`, `threshold_metrics.py` (precision/recall/F1) |
| Table 5: SPECTER2 retrieval audit | pair manifests, `allenai/specter2_base` | `benchmark/bench165/scripts/bench_stats_specter_align.py` |
| Tables 6–7: Agents4Science topics | `pairs.parquet` (`topic`, `secondary_topic`) | `benchmark/construction/pairs_a4s/a10_balance.py` |
| App. B: pairing rule and construction | FARS dump, arXiv, OpenReview, Pangram dump | `benchmark/construction/` (steps `s1`–`s9` for FARS, `a1`–`a11` and `v7_*` for Agents4Science), `benchmark/bench165/scripts/build_views165.py`, `benchmark/benchA4S/scripts/{pdf_to_tex,build_viewsA4S}.py` |
| Detector rows of Table 2 | GPU, falcon-7b, t5-3b, gpt-j-6b, gpt-neo-2.7B | `baselines/detectors/`, runners in `benchmark/bench165/scripts/run165_*.sh`, `benchmark/benchA4S/jobs/det_*.sbatch` |
| Reviewer rows of Table 2 | vLLM servers for CycleReviewer-8B and Qwen2.5-32B | `baselines/reviewers/`, `benchmark/bench165/scripts/bench_reviews.py`, `benchmark/benchA4S/scripts/bench_reviewsA4S.py` |
| Fig. 4, App. E: ICLR 2017–2026 | OpenReview/HF mirrors, Pangram public dump, GPU | `iclr_analysis/` |
| §4, App. C: SciSlopHarness | Claude Code CLI (`claude-haiku-4-5-20251001` editor, `claude-sonnet-5` reviewer), FARS records | `scislopharness/harness/harness.py`, `quality_gate.py`, skill `scislopharness/skill/SciSlop_v0.5.md`, driver `scislopharness/drivers/run_batch_v14.py` |
| App. D: revision baselines | Claude Code CLI, Qwen2.5-32B reviewer | `scislopharness/revision_baselines/run_arm.py` (arms `a1_base`, `a2_code`, `a3_review`, `a4_slop`) |
| Fig. 5, Table 12: distance to the human mean | harness and baseline run trees | `scislopharness/drivers/fig_harness_result.py`, `tab_harness_gap.py` |
| Fig. 6: component ablation | ablation run trees | `scislopharness/ablation/code/run_ablation.py`, `fig_component_summary.py` |
| Table 4, Table 14: matched revision cases | run trees | `scislopharness/drivers/case_study_v2.py`, `casepick.py` |

Run trees, model outputs and caches (several GB) are not in the repository; the scripts that consume them are, with
their expected paths, so a rerun regenerates them in place.

## Data root layout

Every script addresses the working tree by relative path under `$SCISLOP_ROOT` (`scripts/make_root.sh` creates the
code part of that tree as symlinks into this repository). Data that has to be supplied separately:

| Path under `$SCISLOP_ROOT` | Content | How to obtain |
|---|---|---|
| `paper/draft_v6/scislopbench/bench165/`, `benchA4S/` | items + body views of the 390 pairs | `scislopbench/build/materialize_bench.py` from the released Parquet tables |
| `fars/papers/FA????_*/code/writing/paper/main.tex`, `code/exp/EXPERIMENT_RESULTS/` | the FARS dump: 166 generated papers with LaTeX and experiment records | from the FARS authors; needed for the harness and for `run_slop165.py` (the release runner `run_measure_release.py` does not need it) |
| `ana/reference/data/human_refs/eprints/<arxiv>/`, `paper/draft_v6/scislopbench/data/eprints_new/` | arXiv e-prints of the human anchors | `arxiv.org/e-print/<id>`; `s7_download.py`, `a7_download.py` |
| `Agents4Science/2025_full/papers/` | the 247 Agents4Science 2025 PDFs and `metadata.jsonl` | OpenReview (`a4s_openreview_url` in `pairs.parquet`) |
| `artifact-ai2science/Evaluation/ICLR/data/`, `ICLR2026_Pangram/data/` | ICLR 2017–2026 corpora | `iclr_analysis/collect/` |
| `paper/draft_v6/slop/_common/_llm_cache/`, `_pmi_cache/` | cached Qwen label / PMI calls | regenerated on rerun (GPU) |
| `~/.cache/huggingface/` | model weights | Hugging Face Hub |

## Models and serving

All measurement calls use open models served locally: Qwen2.5-32B-Instruct on vLLM (`scislop/_llm/serve_qwen.sh`,
OpenAI-compatible endpoint in `LLM_ENDPOINT`), Qwen2.5-7B-Instruct for PMI through `transformers`, and
Qwen2.5-VL-32B-Instruct for figure transcription; greedy decoding, bfloat16, every call cached by a hash of prompt and
model. Detectors use falcon-7b(-instruct), t5-3b, gpt-j-6b and gpt-neo-2.7B. The reviewer baselines use
`WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B` and Qwen2.5-32B-Instruct. SciSlopHarness and the revision baselines call
Claude models through the Claude Code CLI. SLURM scripts (`*.sbatch`, `run*.sh`) document the GPU shapes used; set
`SLURM_QOS`, `VLLM_PYTHON` and `LLM_ENDPOINT` for your cluster.

## Repository layout

```
scislop/               the six measures: _common/ (views, records, LLM/PMI clients), Structure/, Argument/, Artifacts/, _llm/ (vLLM server)
scislopbench/          SciSlopBench data (data/*.parquet), dataset card, build/ (release builder, materialize_bench.py)
benchmark/             construction/ (pairing pipeline for both halves), bench165/ and benchA4S/ (runners, baselines, tables), run_measure_release.py
baselines/             detectors/ (Binoculars, DetectGPT, Fast-DetectGPT, NTS) and reviewers/ (CycleReviewer, AI Scientist, CMU reviewer)
iclr_analysis/         ICLR 2017–2025 and ICLR 2026 collection, measurement, calibration and Figure 4
scislopharness/        harness/ (core loop and reviewer gate), skill/ (SciSlop v0.5), drivers/ (batch runs, figures, reports), revision_baselines/, ablation/
scripts/make_root.sh   builds the $SCISLOP_ROOT layout
requirements.txt
```

Each folder's `README.md` describes its contents; per-measure `RUN.md` files give the exact invocations used.

## Citation

```bibtex
@inproceedings{scislop2027,
  title     = {Science or Slop?: Benchmarking and Mitigating Scientific Slop in AI-Generated Papers},
  author    = {Anonymous},
  booktitle = {Under review at the International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```
