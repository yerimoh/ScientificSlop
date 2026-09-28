# macro_redund — Run Guide
- **Model**: none. Deterministic n-gram matching over the prose view.
- **Input**: AI = `fars/papers/FA*/code/writing/paper/main.tex` with `\input` expanded in document order; HU = anchor arXiv sources in `Baseline_Sandbagging/results/_human_tex_full/<arxiv>/`. Symmetric loader `_common/views.load_doc`; floats, captions, verbatim boxes and declaration macros removed; math/cite/ref as sentinels; headings and statement sections (ethics, impact, reproducibility, checklist) dropped.
- **Unit**: the sentence, eligible at 8 tokens or more.
- **Measurement**: every 8-gram is indexed by its first occurrence in document order. In a sentence, an 8-gram counts when its first occurrence is in an earlier sentence of a different top-level section; the 8 positions are marked. `r(s)` = marked tokens / tokens. Recycled = eligible and `r(s) >= 0.5`. Score = recycled / eligible; the same share inside body sections and inside summary slots (abstract, conclusion).
- **Metric keys**: `slop_score, coverage, slop_score_agg, slop_body, slop_summary, repeat_word_rate, mean_r, longest_repeat, n_recycled, n_eligible, n_sentences, has_recycled, recycled_by_role, recycled_by_pair`.

## Run
```bash
cd paper/draft_v6/slop/Structure/macro_redund/code
python3 measure.py                              # full census, about two minutes, no GPU -> ../results/
python3 measure.py --only FA0005,2402.06363     # one pair, for the regression checks
python3 measure.py --n 6  --out /tmp/n6         # sensitivity: N (also 10); --tau 0.4 / 0.6 for TAU
python3 measure.py --cliche-min 10 --out /tmp/c # optional stock-phrase filter (8-grams in >= 10 human papers); empty at n = 8
```
Outputs in `../results/`: `papers.jsonl` (per paper), `sentences.jsonl` (every sentence with at least one covered token, with `r`, longest run, slot, and the source sentence that supplied the words), `summary.json` (AI against HU, quantiles, strata by sentence count, recycled sentences by role and role pair, confound audit).

## Regression cases (must hold after any change)
| paper | sentence | expected |
|---|---|---|
| FA0016 | Abstract sentence (36 tokens) repeated in the Introduction and again in the Conclusion | both recycled, `r` = 1.0, slots `body` and `summary` |
| FA0059 | Method: "The key insight is that by filtering out stale versions …" | recycled, slot `body`, `r` about 0.7 |
| FA0005 | Method: "We propose Selective DeLex-JSON, a training-free defense …" | recycled, `r` = 1.0, source = Abstract (first occurrence) |
| 2112.05329 | Experiments: "For each HIT (human intelligence task), the AMT interface shows…" shares one 8-token phrase with the Introduction | `r` about 0.24, not recycled; its near-copy in the appendix ("User Study") is not a unit |
| 2201.09865 | any | no sentence containing `MISSING INPUT` |
| 1902.10416 | any | no sentence made of `\SetKw…` declarations ("Range range KwTo …") |
| 2410.13846, 2210.14140 | subsection headings printed as one-line sentences | not a unit |
| FA0301 (0909) | table rows | no match once floats are removed |
| any (0909) | "Lipman et al 2022" as a repeated citation phrase | no match once `\cite` is a sentinel |

```bash
python3 measure.py --only FA0016,FA0059,FA0005,2112.05329,2201.09865,1902.10416,2410.13846,2210.14140 --out /tmp/reg
```
