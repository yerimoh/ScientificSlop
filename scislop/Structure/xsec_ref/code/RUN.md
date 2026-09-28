# xsec_ref — Run Guide
- **Model**: none. Deterministic LaTeX and prose parsers.
- **Input**: AI = `fars/papers/FA*/code/writing/paper/main.tex` with `\input` expanded in document order; HU = anchor arXiv sources in `Baseline_Sandbagging/results/_human_tex_full/<arxiv>/` (main file = the one carrying `\documentclass`). Symmetric loader `_common/views.load_doc`.
- **Unit**: the declared object. Every top-level body section, plus every label declared inside one, except a label written on the section heading itself. Sub-section labels are objects inside their parent section. Appendix material is not an object.
- **Measurement**: three parsers produce pointer events. K1 reference macros (`\ref`, `\cref`, `\autoref`, `\Secref`, `\twosecrefs` ...), K2 printed section numbers in prose, K3 deictic and named section references. Reach of an object is the number of distinct sections other than its home that point to it. Roadmap events in the Introduction are excluded from reach and reported as a variant.
- **Metric keys**: `slop_score, coverage, reuse_rate, referenced_rate, mean_reach, max_reach, reuse_rate_by_kind, reuse_rate_by_home_role, backward_links, forward_links, reuse_rate_incl_roadmap, appendix_pointers, events_by_parser, unresolved_reasons, has_reuse`.

## Run
```bash
cd paper/draft_v6/slop/Structure/xsec_ref/code
python3 measure.py                 # full census, about one minute, no GPU
python3 measure.py --only FA0001   # single paper, for the regression checks
```
Outputs in `../results/`: `papers.jsonl` (per paper), `objects.jsonl` (per object, with home, kind, reach and the sections that reached it), `summary.json` (AI against HU, reuse by kind, strata by object count, parser contributions, confound audit).
