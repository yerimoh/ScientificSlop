> Working note translated from the authors' Korean notes; the folder README.md is the authoritative description.
# Human-pair selection rule (SciSlopBench) — finalized 0911

This is the procedure for attaching one human-written paper to each FARS paper. It was derived by applying it to 5 FARS papers (`PAIRS_README_0911_v3.md`) and is used unchanged for the 165-paper expansion. The implementation is `scripts/build_pairs5_v3.py`, and the candidate table is kept in the `candidates_5fars_0911_v3.csv` format.

The principle in one line. **Pick a paper of the same kind that addresses the same problem in the same way, but it must be provable that a human wrote it and a conference accepted it.**

---

## Step 0. Label the type of the FARS paper first

Read the title, abstract and section structure and assign one of six labels.

| Type | Basis for the verdict |
|---|---|
| method | Proposes a new method, algorithm or pipeline and has a method section |
| framework | Proposes an evaluation/analysis framework or a metric system |
| benchmark | Builds a task, data or leaderboard and has a data-construction section |
| dataset | The data itself is the contribution |
| analysis | Measures and dissects a phenomenon without proposing a new method |
| survey | Organizes prior work |

This label determines which anchors qualify. All 5 current papers are of the method type.

---

## Step 1. Build the candidate pool (papers cited by the FARS paper)

1. Split the tex by section and collect the `\cite` keys per section. Section names are normalized to intro / related / method / experiments / conclusion.
2. Resolve keys to arXiv ids. **Do not trust only the arXiv id in the bib entry.** Entries without an id are filled by looking up arXiv by title. Across the 5 papers this brought in 8 additional papers, one of which became a final anchor.
3. Fetch the original directory structure from `arxiv.org/e-print/<id>`. Flattened tex is not used.
4. If no candidate is found, widen to uncited papers on the same topic, but if even one cited candidate passes, uncited ones are not used.

---

## Step 2. Apply the four hard conditions (all must pass to be a candidate)

| # | Condition | How judged | Why |
|---|---|---|---|
| H1 | Human authorship verified | If an ICLR 2026 submission, Pangram AI fraction ≤ 0.05; otherwise arXiv v1 year ≤ 2024 | If the negative class is AI-assisted papers, the benchmark itself collapses. In practice AgentFold 0.31, FaithCoT-Bench 0.36 and DeepTRACE 0.10 were caught |
| H2 | Accepted to a top-tier main track | Confirmed via arXiv comment, journal_ref or OpenReview record. Rejected-only, workshop, preprint and journal-survey papers are excluded. Later acceptance at a different venue is recognized | The task assumption that reviewers give human papers higher scores holds only for accepted papers |
| H3 | Type match | Anchor type = the FARS type from Step 0 | Benchmark papers have data/metric sections without a method section, and surveys differ by an order of magnitude in citation count. If types differ, the denominator of structural slop differs and the comparison does not hold |
| H4 | Manuscript usable | After locating the root tex and expanding `\input` in document order: body ≥ 2,500 words, ≥ 4 sections, body and appendix separated | In a manuscript whose section order has collapsed, all structural measurements are void |

Applied to 64 cited candidates, 20 remained. Rejection reasons: type mismatch 34, not top-tier 21, human authorship unverifiable 14, manuscript unusable 2 (with double counting). **The type condition filters out the most.** This is because the benchmarks and datasets the FARS paper used in its experiments account for half of the citations.

---

## Step 3. Sort the passing candidates in this order

### Priority 1. Topic-similarity tier

Assigned by reading the title and abstract. SPECTER2 cosine is used as a tie-break only within the same tier.

| Tier | Definition | Example |
|---|---|---|
| 3 | Same problem + same kind of contribution | QuoteVerify and RARR are both inference-time pipelines that attach evidence to generated output post hoc and fix the unsupported parts |
| 2 | Same problem, different kind of contribution | Qi 2023 addresses safety degradation from fine-tuning, but as an attack analysis rather than a defense |
| 1 | Same field, different problem | Reflexion is agent memory while FARS is context compression |
| 0 | Cited only as a tool, model or data | Qwen2.5 technical report, MMLU |

### Priority 2 and below

| # | Criterion | Notes |
|---|---|---|
| 2 | Citation relation. Cited in experiments (inheriting setting/baselines) = parent > cited in related work/intro | The parent passed the four hard conditions in only 1 of the 5 papers |
| 3 | Has review scores | ICLR 2026: `Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl`; ICLR 2025: HF dataset `QAQqaq/ICLR2025Openreview` |
| 4 | Body word ratio (human/AI) ≤ 2.5 | Passing pairs are 1.57–2.35. Since FARS is a 4-page report, 1.0 does not occur |
| 5 | Public code repository + method diagram in the body | Needed to measure code_paper and fig-family slop on both sides with the same rule |

---

## Step 4. Record and publish

- Keep rank 1 as primary and ranks 2–3 as alternates.
- Record the most similar candidate blocked by a hard condition together with the condition that blocked it. This is the answer to "there was a better match, so why wasn't it used".
- Include a balance table (body words, appendix, sections, figures, tables, unique citations, internal references, template) together with the **AUROC of a classifier that uses only trivial features** (length, number of references, number of internal references). If we do not measure it first, a reviewer will measure it for us.
- If no candidate passes the four hard conditions, drop that FARS paper or use the closest one while recording the violated condition in the manifest.
- Slop metrics are used only as ratios, never as counts (relative to references, tables, numbers).

---

## Criteria we decided not to use

| Criterion | Why rejected |
|---|---|
| Title similarity | FARS titles are mostly coinages, so TF-IDF is 0.00 for most candidates, and what scores high is the name of a benchmark the FARS paper used in its experiments |
| Embedding cosine alone | All candidates cluster at 0.90–0.97 and the order flips on 0.01 differences. Used only as a tie-break within a tier verdict |
| Citation count / fame | Tool citations such as InstructGPT, LoRA and the Qwen technical report rise to the top |
| ICLR-oral constraint | The era is off by 2+ years and the type skews toward benchmarks. The old PoC anchors had this problem |
| Only papers with scores | Becomes ICLR-only and forces giving up type match. Scores are placed at priority 4 |
| Contemporaneous 18-month window | Conflicts with the citation pool. The parents of FARS papers are from the second half of 2025, and there is no means of verifying human authorship for that period |

---

## Known remaining issues

1. For 2025 papers that are not ICLR 2026 submissions there is no way to verify human authorship. This blocked four parent papers (In-Training Defenses, Context as a Tool, ReportBench, AgentRewardBench). If Pangram can be run again, H1 is resolved.
2. Human-side review scores exist only for ICLR. NeurIPS/ACL anchors have no scores, so only 3 of the 5 pairs can be used for score correlation.
3. The similarity tier is a single judgment with no second judge. The tier table is kept in the script so it can be re-judged.
4. The body ratio remains at 1.5–2.4. This is a structural difference that does not go away even when the human paper is cut to its body only.
