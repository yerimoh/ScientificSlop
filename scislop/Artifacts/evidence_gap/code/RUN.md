# evidence_gap — Run Guide
- **Model**: none. Deterministic (display environments, example captions, long quotes outside Related Work; section spans for the titled-without-exhibit sub-observation).
- **Input**: AI = FARS papers with LaTeX; Human = anchor arXiv sources (`_common/corpus.py`). Applicable = papers with at least one body result table; others NA.
- **Metric keys**: per paper `n_exhibits, exhibit_kinds, exhibits_per10k, exhibits_appendix_only, titled_without_exhibit, slop_score (1 = no exhibit anywhere), coverage`; per instance `verdict` in {exhibited, titled_without_exhibit, unexhibited}.

## Run
```bash
cd paper/draft_v6/slop/Artifacts/evidence_gap/code
python3 measure.py                 # both corpora
python3 measure.py --corpus AI     # FARS only
```
Outputs in `../results/`: `instances.jsonl`, `papers.jsonl`, `summary.json`.
Reading rules: observational, no paper label; report the human base rate next to the AI rate; appendix-only exhibits are counted and flagged, never discounted; a safety paper's unexhibited verdict carries the deliberate-withholding note before any use.
