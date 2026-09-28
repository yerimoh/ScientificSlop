# SciSlopBench: construction and evaluation code

The data itself (390 pairs, 773 paper bodies, per-side scores) is in [`../scislopbench/`](../scislopbench/). This
folder holds the code that built it (Appendix B) and that evaluates systems on it (Tables 2–3, 5).

```
construction/
  PAIR_RULE_0911.md          the pairing rule as applied (H1–H4, tiers, ranking); Appendix B.1
  common/slopbench_lib.py    TeX flattening in document order, body/appendix split, structural metrics behind H4
  common/build_pairs5*.py    the 5-pair proof of concept that seeded the rule (frozen into the FARS half)
  pairs_fars/                FARS half, steps s1–s9 (143 pairs)
  pairs_a4s/                 Agents4Science half, steps a1–a11 and v7_* (247 pairs)
bench165/scripts/            FARS half: view builder, measure/detector/reviewer runners, statistics
benchA4S/scripts/, jobs/     Agents4Science half: PDF→TeX, view builder, runners, figure-exposition pipeline, tables
run_measure_release.py       runs one measure on either half from the released views (no FARS dump needed)
```

## Construction pipeline (Appendix B)

Both halves follow the same steps; the FARS scripts are `s<n>_*.py` in `construction/pairs_fars/`, the
Agents4Science scripts `a<n>_*.py` in `construction/pairs_a4s/`. Outputs go to a `cache/` next to the scripts.

| Step | FARS | Agents4Science | External resource |
|---|---|---|---|
| Index the AI papers (sections, cites per section, figures, tables) | `s1_fars_index.py` reads the FARS dump | `a1_a4s_index.py` reads the OpenReview PDFs with PyMuPDF | – |
| Cited candidate pool | `s2_pool.py`, `s2b_merge_titles.py` | `a2_pool.py` | – |
| Resolve titles to arXiv ids, fetch metadata | `s3_arxiv_fetch.py` | `a3_resolve.py` (OpenAlex, then arXiv), `a3b_meta.py` | arXiv API, OpenAlex |
| H4 source metrics on candidate e-prints | `s4_cand_metrics.py` | `a4_cand_metrics.py` | e-prints on disk |
| H1 (year ≤ 2024 or ICLR 2026 with Pangram share ≤ 0.05), H2 (accepted, top-tier), heuristic H3 screen, judge batches | `s5_shortlist.py` | `a5_shortlist.py` | Pangram shares from the public ICLR 2026 dump (`iclr_analysis/collect/iclr_2026`) |
| Contribution type and similarity tier | judged from title and abstract with the prompt `JUDGE_INSTRUCTIONS.md`; outputs in `cache/judge_out/` | same | LM judge |
| Final H3, ranking (tier, experiment-section citation, public ratings, length ratio ≤ 2.5, SPECTER2), one-to-one assignment | `s6_rank_assign.py` | `a6_rank_assign.py` | `allenai/specter2_base` |
| Download the assigned e-prints, recheck H4 | `s7_download.py` | `a7_download.py` | arXiv e-print |
| Widened pool: accepted ICLR 2026, arXiv id, Pangram ≤ 0.05, TF-IDF top-3 | `s8_widen.py`, `s9_widen_assign.py` | `a8_widen.py`, `a9_widen_assign.py` | – |
| Relaxed fill for topics without a same-type anchor (Agents4Science only) | – | `a10_fill.py` (records `match_rank`), `a11_flag.py` (caveats) | – |
| LaTeX for the AI side | shipped by FARS | `v7_extract_tex.py` (18 supplementary sources), `benchA4S/scripts/pdf_to_tex.py` (229 rebuilt from PDF), `v7_pick_root.py` (≥ 85 % of 40 sampled PDF sentences must be recovered), `v7_build.py` | – |
| Views (`body.tex`, `body.txt`) and items | `bench165/scripts/build_views165.py` | `benchA4S/scripts/build_viewsA4S.py` | – |

The manifests these scripts produce (`pairs165_draft.json`, `pairs_a4s.json`) are what `pairs.parquet` was exported
from, so every pool, tier, condition outcome and rank in the released table can be traced to a step above.

## Evaluation (Tables 2, 3, 5 and the appendix tables)

| Script | Computes | Paper |
|---|---|---|
| `benchA4S/scripts/table_pooled.py` | PairAcc, AUROC, TPR@5 % FPR over the pooled 390 pairs for the six measures, their aggregate, the detectors and the reviewers | Tables 2, 3 |
| `benchA4S/scripts/threshold_metrics.py` | precision / recall / F1 at the 5 % FPR threshold | appendix |
| `benchA4S/scripts/table_numbers.py`, `scorecard.py`, `baselines_scorecard.py` | the same metrics per half | Appendix B (per-source table) |
| `bench165/scripts/bench_stats_specter_align.py` | SPECTER2 retrieval audit (mean cosine, rank of the assigned anchor) | Table 5 |
| `bench165/scripts/bench_stats.py` | benchmark statistics (lengths, sections, figures, tables per side) | §3.2 |
| `benchA4S/scripts/strata.py` | metrics by match rank and by AI-source (shipped vs. rebuilt TeX) | robustness |
| `benchA4S/scripts/length_control.py` | metrics after residualizing every score on log body length | length control |
| `benchA4S/scripts/seed_table.py` | spread over three draws of DetectGPT and the two reviewers | seed variance |
| `bench165/scripts/aggregate165.py`, `benchA4S/scripts/compare_a4s.py` | per-half aggregation helpers | – |

`table_pooled.py` reads `results/slop/<measure>/papers.jsonl`, `results/<detector>.jsonl` and
`results/reviews/<system>/*.json` under both benches; `run_measure_release.py`, the detector runners and the reviewer
runners write exactly those files.

## Runners

| What | FARS half (`bench165/scripts/`) | Agents4Science half (`benchA4S/`) |
|---|---|---|
| Measures | `run_slop165.py --checker <m>` (needs the FARS dump) or `../run_measure_release.py --half fars` | `scripts/run_slopA4S.py --checker <m>` or `../run_measure_release.py --half a4s` |
| Argument graph (labels, then PMI) | `run165_ag.sh`; ablations `run165_ag_judge.sh`, `run165_ag_pmi.sh` | `jobs/ag_labels.sbatch`, `jobs/ag_pmi*.sbatch` |
| Figure exposition | `scislop/Artifacts/fig_exposition/code/measure.py` | `scripts/figexp_candidates.py` → `figexp_gate.py` → `figexp_prepare.py` → `run_figexp_a4s.py` (`jobs/figexp_vlm.sbatch`) |
| Detectors | `run165_detectors.sh`, faithful settings `run165_bino_faithful.sh`, `run165_dgpt_faithful.sh` | `jobs/det_a.sbatch` (Binoculars, NTS, Fast-DetectGPT), `jobs/det_b.sbatch` (DetectGPT) |
| Pangram | `scripts/pangram_bench.py` (paid API; key from `PANGRAM_API_KEY`) | same script |
| Reviewers | `bench_reviews.py` with `run165_b2h*.sh` (CycleReviewer) and `run165_qwen*.sh` (Qwen server for AI Scientist) | `scripts/bench_reviewsA4S.py --run/--shard`, `jobs/rev_*.sbatch` |
| Everything, in order | `run_all165.sh`, `finish_*.sh` (supervisors that wait for servers and resume) | `jobs/llm_all3.sbatch` |

Shell and SLURM scripts document the exact GPU shapes and orderings used; they reference `SLURM_QOS`, `VLLM_PYTHON`
and `LLM_ENDPOINT` from the environment.

