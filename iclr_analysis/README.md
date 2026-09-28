# ICLR analysis: slop against review scores and decisions (§5.2, Fig. 4, Appendix E)

Does scientific slop mark provenance, quality, or both? This folder measures every system of Table 2 on ICLR
submissions with public reviews and relates the scores to ratings and decisions.

Two corpora (Appendix E):

| Corpus | Papers | Use | Collection |
|---|---|---|---|
| ICLR 2017–2025 | 1,381 decided papers with LaTeX source, 1,177 after the quality audit (590 rejected, 434 accepted, 153 oral) | Fig. 4a (AI likeness vs. rating 2–9), Fig. 4c (rejected-vs-accepted AUROC within each year) | `collect/iclr_2017_2025/` |
| ICLR 2026 | 691 papers (560 accepted, 131 rejected), sampled over Pangram-share bands × decision | Fig. 4b (Spearman ρ with Overall, Soundness, Presentation, Contribution) | `collect/iclr_2026/` |

## Collection

`collect/iclr_2017_2025/`: `collect_iclr_dataset.py` (OpenReview v1 API for 2017–2023: decisions, reviews, arXiv
e-print and PDF), `collect_strata.py` (year × rating and year × award strata), `build_labels_2425.py` +
`collect_2425.py` (2024–2025 from Hugging Face mirrors of OpenReview), `extend_scores_0915.py` (top-up matched through
Semantic Scholar / OpenAlex / arXiv), `replace_from_mirror.py`, `replace_texless.py`, `retry_tex.py`, `enrich_meta.py`,
`oai_harvest.py`, `verify_integrity.py`; `ocr_pdf.py` (Mistral OCR, `MISTRAL_API_KEY`) for papers without source.
Data lands in `$SCISLOP_ROOT/artifact-ai2science/Evaluation/ICLR/data/<year>/<note_id>/{meta.json,tex/,paper.pdf}`.

`collect/iclr_2026/`: `build_dataset.py` joins the public Pangram verdicts of the ICLR 2026 detection partnership
(`iclr.pangram.com/data/{submissions,reviews}.json`, `fraction_ai` per submission) with the accept flags and arXiv ids
of the `ai-conferences/ICLR2026` mirror into `papers_2026.jsonl`; `collect_texts.py` fetches the arXiv sources of the
stratified sample; `pangram_api.py` is the Pangram v3 client (`PANGRAM_API_KEY`); `analyze.py` and `plots.py`
summarize the 19k-submission corpus. `papers_2026.jsonl` is also the Pangram source of pairing condition H1 and of the
widened anchor pool in `../benchmark/construction/`.

## Measurement and analysis (`scripts/`)

Run in this order (the shell scripts `pipeline_v2.sh`, `pipeline_after_crawl.sh` chain them on SLURM):

1. `build_iclr_records.py`, `build_iclr2026_records.py` → one record per paper (`records/*.json`)
2. `build_iclr_texts.py`, `build_texts_2026.py` → body and prose views
3. `quality_audit.py` → exclusions (extraction failures, fewer than 3 reviews, body length outside the 1–99 percentile)
4. Systems, all sharing the loader `_systems.py`:
   - SciSlop measures: `run_slop_iclr.py` (Macro redundancy, Cross-section references, Citation isolation, Evidence gap, Argument graph; the Argument-graph server jobs are `ag_*.sh`), `figexp_*.py/.sh` for Figure exposition (PDF fetch, method-figure pick, crop, VLM transcription, score)
   - detectors: `run_detectors_iclr.sh`, `run_detectors_2026.sh` (Binoculars, DetectGPT, Fast-DetectGPT, NTS from `../baselines/detectors/`)
   - reviewers: `run_reviews_iclr.py` with `rev_*.sh` / `rev26_*.sh` on the subsets drawn by `make_subset_*.py` (CycleReviewer 287 papers, AI Scientist 282 on 2017–2025; 338 and 285 on 2026)
5. `export_scores.py` → `scores_years.csv`, `scores_2026.csv`
6. `calibrate_scores.py` → a logistic fit of each system on the 143 FARS pairs, applied unchanged to ICLR, giving a
   calibrated AI probability; year-adjusted by centring on the year mean (`normalize_scores.py`)
7. `fig_scale_points.py` (panel a: probability vs. rounded rating), `fig_dimensions.py` (panel b: Spearman ρ on ICLR 2026),
   `fig_review_alignment.py` (draws the three panels and the legend, `fig_review_alignment_{a,b,c,legend}.pdf`; panel c is the
   within-year rejected-vs-accepted AUROC, drawn for systems with at least 15 papers per side in at least five years)
8. `table_review_year_auroc.py` → the per-year AUROC table of Appendix E; `make_workbook.py` → an .xlsx of all scores

The remaining `fig_*.py` files (`fig_review_3panel*`, `fig_final_review*`, `fig_stairs*`, `fig_rating_*`,
`fig_year_highlow.py`, `fig_lines.py`) are earlier drafts of the same figure kept for reference; `analyze_*.py` are the
summaries behind the text of Appendix E.
