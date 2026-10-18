# thin_grounding — Run Guide (v1)

- **Model**: none. Deterministic, reads `\cite` keys and the prose view of the body.
- **Input**: the expanded tex through `_common/views.load_doc`; body sections only (appendix and acknowledgements excluded).
- **Measurement**: per top-level body section of at least 80 prose words, whether it cites any work. `slop_score = uncited sections / body sections`, weak below 3 sections. Graded observation `works_per_1k` = distinct cited works in the body per 1,000 body prose words.
- **Metric keys**: `slop_score, slop_numerator, slop_denominator, weak, coverage, n_body_sections, n_uncited_sections, n_works, n_citations, body_words, works_per_1k, citations_per_work, no_cite_commands, raw_markers_outside_cite, extraction_suspect, sections`.

## Run
```bash
python3 benchmark/run_measure_release.py --half fars --checker thin_grounding
python3 benchmark/run_measure_release.py --half a4s  --checker thin_grounding
# or, from this folder, on the corpus registry
python3 measure.py [--corpus AI|HU|all] [--only IDS]
```
Outputs in `../results/` (or the runner's results folder): `papers.jsonl`, `instances.jsonl` (one record per body section), `summary.json`. About a second per paper.

## Regression cases (must hold after any change)
| case | expected |
|---|---|
| `\citep{a,b}` then `\citet{a}` in another section | `n_works` = 2, `n_citations` = 3 |
| section under 80 prose words | not a unit |
| appendix section with citations | not counted |
| body with no `\cite` command | `no_cite_commands` = true, `extraction_suspect` = true |
| "(Smith et al., 2020)" left as plain text three times | `raw_markers_outside_cite` >= 3, `extraction_suspect` = true |
