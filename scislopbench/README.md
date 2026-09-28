# SciSlopBench

SciSlopBench is a paired benchmark for **scientific slop**: failures of global scientific reasoning in AI-generated
research papers that token-level AI detectors do not see. It holds **390 pairs**. Each pair is one AI-generated paper
and one human-written, peer-accepted paper (its *anchor*) of the same contribution type on the same or a neighboring
research problem. The AI papers come from two sources, and the same pairing rule is applied to both:

| Half | AI papers | Anchor pool | Topic similarity tier 3 / 2 / 1 / 0 |
|---|---|---|---|
| FARS | 143 papers generated end to end by the FARS system | 120 cited by the AI paper, 23 from accepted ICLR 2026 | 93 / 23 / 27 / 0 |
| Agents4Science 2025 | all 247 submissions to the Agents4Science 2025 conference | 66 cited, 119 from accepted ICLR 2026, 62 nearest admissible anchor | 63 / 133 / 45 / 6 |

Contribution types over the 390 pairs: method 286, analysis 71, framework 21, benchmark 6, survey 6.
Topics: the FARS half covers eight machine-learning areas; the Agents4Science half is mostly computer and data
sciences (178) with life and health sciences (17), social sciences (14), natural sciences (13), interdisciplinary (11),
engineering (6), mathematics and statistics (4), humanities (2), and law, policy and business (1).

The benchmark accompanies the paper *Science or Slop? Benchmarking and Mitigating Scientific Slop in AI-Generated
Papers* (under review, ICLR 2027). The paper's six SciSlop measures across Structure, Argument and Artifacts identify the AI
paper in 85.9% of pairs, against 68.7% for Binoculars. All numbers of the paper's benchmark tables are recomputable from
`scores` alone (see below).

## What is in the release

Three tables, all Parquet, under `data/`:

```python
import pandas as pd
pairs  = pd.read_parquet("data/pairs.parquet")    # 390 rows, one per pair
papers = pd.read_parquet("data/papers.parquet")   # 773 rows, one per unique paper
scores = pd.read_parquet("data/scores.parquet")   # 780 rows, one per pair side
```

or with the `datasets` library, `load_dataset("parquet", data_files="data/pairs.parquet")`.

### `pairs` (390 rows): the pairing record

Every match can be checked independently: the candidate pool, similarity tier, the four anchor conditions and the
retrieval rank are stored for each pair.

| Column | Meaning |
|---|---|
| `pair_id` | `FA0001` … (FARS) or `A4S0011` … (Agents4Science) |
| `source` | `fars` or `agents4science` |
| `ai_paper_id`, `human_paper_id` | keys into `papers` (`AI_<pair_id>` and `HU_<arxiv id>`) |
| `ai_title`, `human_title`, `human_arxiv`, `human_venue`, `human_v1_year` | identity of both sides; the anchor is always an arXiv e-print |
| `human_iclr_rating` | mean public ICLR review rating of the anchor when it exists (221 anchors) |
| `contribution_type`, `human_contribution_type`, `same_contribution_type` | one of method, framework, benchmark, dataset, analysis, survey, judged from title and abstract |
| `pool` | `cited` (anchor cited by the AI paper), `widened_iclr26` (accepted ICLR 2026 paper found by TF-IDF), `filled_iclr26` (Agents4Science only: closest admissible anchor when no same-type anchor exists) |
| `sim_tier` | 3 same problem and contribution type, 2 same problem, 1 same area, 0 cited only as a tool, model or dataset |
| `relation`, `tier_reason` | how the anchor relates to the AI paper; the judge's one-line reason (Agents4Science half) |
| `H1_human`, `H1_by`, `H2_top_tier`, `H3_type_match`, `H4_source` | outcome of the four anchor conditions (see *Pairing rule*); `H1_by` is `year<=2024` or `pangram` |
| `specter2_cosine`, `tfidf_similarity`, `match_rank` | retrieval diagnostics (SPECTER2 cosine for FARS, TF-IDF for Agents4Science; rank of the assigned anchor among candidates) |
| `human_pangram_ai_fraction` | Pangram AI share of the anchor when H1 was checked with Pangram (≤ 0.05 by construction) |
| `ai_body_words`, `human_body_words`, `body_length_ratio` | body lengths of the two sides |
| `topic`, `secondary_topic` | FARS: one of eight ML areas; Agents4Science: the submission's primary and secondary topic |
| `a4s_submission_id`, `a4s_openreview_url`, `a4s_decision`, `a4s_reviewer_scores`, `a4s_ai_involvement` | Agents4Science metadata: OpenReview id and URL, accept/reject decision (48 / 199), the three AI reviewer scores and the human score as JSON, and the self-reported AI involvement per stage (A–D) as JSON. These are metadata only and were not used for pairing |
| `ai_tex_source` | `shipped_tex` (LaTeX provided by the system or in the supplementary material, 161 papers) or `rebuilt_from_pdf` (LaTeX reconstructed from the OpenReview PDF, 229 papers) |

Evaluation subsets used in the paper, expressed on `pairs`:

- **all**: 390 pairs;
- **problem-matched**: `sim_tier >= 2` (116 FARS + 196 Agents4Science = 312 pairs);
- **strict**: `source == "fars"` or `pool != "filled_iclr26"` (143 + 185 = 328 pairs, same contribution type and a cited or widened-pool anchor).

### `papers` (773 rows): the two document views

One row per unique paper. Seven human anchors serve one FARS pair and one Agents4Science pair each and appear once,
with both pairs listed in `pair_ids`.

| Column | Meaning |
|---|---|
| `paper_id` | `AI_<pair_id>` or `HU_<arxiv id>` |
| `pair_ids` | list of the pairs this paper belongs to |
| `source` | `fars`, `agents4science`, or `fars+agents4science` for the seven shared anchors |
| `label`, `role` | 1 / `ai` for the AI-generated paper, 0 / `human` for the anchor |
| `title`, `arxiv`, `venue` | identity; `arxiv` and `venue` are set for anchors |
| `body_words` | words in the body view |
| `body_tex` | **structure view**: the LaTeX source expanded in document order (`\input` resolved) and cut at the first of appendix, acknowledgements or bibliography. Sections, floats, equations, labels and `\ref`/`\cite` commands are intact, which is what the Structure and Artifacts measures read |
| `body_txt` | **prose view**: the same body as running text, with inline and display mathematics replaced by `MATH`, citations by `CITE` and cross-references by `REF`, which is what the Argument measures and the token-level detectors read |

The body view is the paper without its appendix so that AI and human papers are compared on the part every reader
sees; the human anchors pass H4 only if the appendix is separable and the body has at least 2,500 words in four or more
sections.

### `scores` (780 rows): measures and baselines per pair side

One row per side of each pair (`pair_id`, `paper_id`, `label`). The SciSlop measures are deterministic per paper, so
the seven shared anchors carry the same measure scores in both of their rows; the detector and LLM-reviewer baselines
were run once per half and are stored as run.

| Column | Direction | Meaning |
|---|---|---|
| `scislop_macro_redund`, `scislop_xsec_ref` | higher = more slop | Structure: **Macro redundancy**, later sections repeating earlier material instead of developing the argument (cross-section n-gram overlap); **Cross-section references**, sections and labeled objects that no other section refers to |
| `scislop_argument_graph`, `scislop_citation` | higher = more slop | Argument: **Argument graph**, claims that appear before their supporting context; **Citation isolation**, prior work cited without its role in the argument (builds on, differs from, compares against) |
| `scislop_fig_exposition`, `scislop_evidence_gap` | higher = more slop | Artifacts: **Figure exposition**, method diagrams cluttered with experimental details or result claims; **Evidence gap**, aggregate result tables with no concrete input, output or case anywhere in the paper. `null` where the measure does not apply |
| `scislop_aggregate` | higher = more slop | mean over the three planes of the mean of the plane's available measures |
| `binoculars` | lower = more AI-like | Binoculars score |
| `detectgpt`, `fast_detectgpt`, `nts` | higher = more AI-like | DetectGPT, Fast-DetectGPT and the NTS curvature score |
| `pangram` | higher = more AI-like | Pangram AI probability (Pangram 3, v3.3.2). Pangram is also the H1 filter for anchors written after 2024, so it is not an independent detector on this benchmark |
| `cyclereviewer_overall`, `ai_scientist_overall` | lower = more AI-like | overall rating given by the CycleReviewer 8B and AI Scientist reviewer baselines |

Reproducing the paper's pair accuracy (the AI paper scores higher than its anchor; ties count 0.5):

```python
import pandas as pd
s = scores
def pair_accuracy(col, direction=+1):
    w = s.pivot_table(index="pair_id", columns="label", values=col).dropna()
    d = (w[1] - w[0]) * direction
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d), len(d)
pair_accuracy("scislop_aggregate")        # (0.859, 390)
pair_accuracy("binoculars", direction=-1) # (0.687, 390)
```

Table 2 of the paper, recomputed from `scores` (pair accuracy / AUROC over the 390 pairs):

| Method | PairAcc | AUROC | | Plane | Measure | PairAcc | AUROC |
|---|---|---|---|---|---|---|---|
| Binoculars (prose view) | 0.687 | 0.683 | | Structure | Macro redundancy | 0.723 | 0.727 |
| DetectGPT (prose view) | 0.638 | 0.623 | | | Cross-section references | 0.905 | 0.895 |
| NTS (prose view) | 0.626 | 0.607 | | Argument | Argument graph | 0.586 | 0.584 |
| CycleReviewer (full text) | 0.685 | 0.689 | | | Citation isolation | 0.793 | 0.782 |
| AI Scientist reviewer (full text) | 0.615 | 0.613 | | Artifacts | Figure exposition | 0.809 | 0.786 |
| **SciSlop, mean of planes** | **0.859** | **0.854** | | | Evidence gap | 0.764 | 0.765 |

## Pairing rule

A human paper can anchor an AI paper only if it passes four conditions.

- **H1 Human authorship.** Its first arXiv version is dated 2024 or earlier, or it is an accepted ICLR 2026 paper whose Pangram AI share is at most 0.05.
- **H2 Acceptance.** It was accepted to the main track of a top-tier venue, shown by the venue record, the arXiv comment or journal reference, or OpenReview.
- **H3 Contribution type.** It is the same kind of paper as the AI paper (method, framework, benchmark, dataset, analysis or survey).
- **H4 Usable source.** Its arXiv e-print expands in document order to a body of at least 2,500 words in at least four sections, with the appendix separable from the body.

Within each contribution type, the anchor is the candidate closest in research problem. A language model assigns a
topic-similarity tier from title and abstract (3, 2, 1, 0 as above); ties are broken by experiment-section citation,
availability of public review scores, a human-to-AI body-length ratio within 2.5, and finally SPECTER2 cosine. Two
candidate pools are searched in order: papers cited by the AI paper, then 2,690 accepted ICLR 2026 papers with an
arXiv identifier and Pangram share at most 0.05 (three nearest by TF-IDF over title and abstract). Assignment is
one-to-one within each half; contested anchors go to the closer match. For the 62 Agents4Science submissions whose
topics have no same-type anchor in either pool (physics, astronomy, biology, medicine, finance, HCI, …), H1, H2 and H4
are kept and the closest admissible anchor is taken in a recorded order (`pool = filled_iclr26`, `H3_type_match = relaxed`).

## Document processing

- **FARS papers** ship their LaTeX; the body view is built from the shipped source.
- **Agents4Science papers** were collected from OpenReview. 18 provide LaTeX in the supplementary material; for the
  other 229 the LaTeX was reconstructed from the PDF and accepted only if at least 85% of forty sampled PDF sentences
  are recovered verbatim in the rebuilt source.
- **Human anchors** are arXiv e-prints, expanded in document order from the e-print's root file.

## Provenance and licensing

- The pairing record, the scores, the FARS-generated papers and the document views produced for this benchmark are
  released under CC BY 4.0.
- The Agents4Science 2025 submissions are public on OpenReview (`a4s_openreview_url`); copyright stays with their
  authors and systems. Their decisions, reviewer scores and AI-involvement declarations are reproduced as published.
- The human anchors are published arXiv e-prints. Copyright stays with their authors and the text is redistributed
  here, for research on AI-generated scientific writing, under the terms of each e-print's arXiv license. If you are
  an author and want your paper's text removed, open an issue on this repository; the pairing record and scores
  will keep the arXiv identifier so the benchmark stays reproducible.

## Citation

```bibtex
@inproceedings{scislop2027,
  title     = {Science or Slop?: Benchmarking and Mitigating Scientific Slop in AI-Generated Papers},
  author    = {Anonymous},
  booktitle = {Under review at the International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```
