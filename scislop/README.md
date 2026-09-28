# SciSlop: the six measures

Implementation of the six scientific-slop measures of §3.1 / Table 1 / Appendix A. Each measure is one script,
`<plane>/<measure>/code/measure.py`, that reads a paper through the shared loader in `_common/` and writes one JSON
line per paper (`papers.jsonl`) with the fields of Appendix A: `slop_score`, `slop_numerator`, `slop_denominator`,
`coverage`, and the per-unit evidence records.

| Plane | Measure (paper name) | Folder | Unit | Model |
|---|---|---|---|---|
| Structure | Cross-section references | `Structure/xsec_ref` | body sections and labeled objects (figure, table, equation, algorithm, theorem) | – |
| Structure | Macro redundancy | `Structure/macro_redund` | sentences of ≥ 8 tokens in the prose view | – |
| Argument | Argument graph † | `Argument/Argument_Graph` | key claims of the introduction | Qwen2.5-32B-Instruct (labels, majority of 3 greedy runs) + Qwen2.5-7B-Instruct (PMI) |
| Argument | Citation isolation | `Argument/citation` | citing sentences in Introduction and Related Work | – (an optional LLM layer, `--claims`, is not used for the paper's score) |
| Artifacts | Figure exposition † | `Artifacts/fig_exposition` | the method figure | Qwen2.5-VL-32B-Instruct (transcription only) |
| Artifacts | Evidence gap | `Artifacts/evidence_gap` | the paper (applies when the body has a result table with ≥ 2 data rows and ≥ 4 numeric cells) | – |

Score = failed units / checked units, higher = more slop. The aggregate of Eq. 7 is the mean of the measures within a
plane, then the mean of the planes (implemented in `_common/records.py::aggregate_scores` and in
`scislopbench/build/build_hf_release.py`).

## Shared code (`_common/`)

| File | Role |
|---|---|
| `views.py` | LaTeX expansion in document order (`\input`/`\include`), section detection and role map (`section_role_map.tsv`), body/appendix split, the prose view with `MATH` / `CITE` / `REF` sentinels, sentence splitting, reference events |
| `corpus.py` | paper registry (`ai_papers()`, `hu_papers()`); the benchmark runners replace these two functions with records built from the items file |
| `records.py` | evidence records, `slop_score`, `aggregate_scores`, Cliff's delta, AUROC |
| `tables.py` | LaTeX table parser used by Evidence gap |
| `llm.py` | cached JSON calls to the Qwen server (cache in `_llm_cache/`, key = hash of prompt and model tag `MODEL_TAG`), verbatim quote verification |
| `lm_score.py` | PMI log-probabilities with `transformers` (`PMI_MODEL`, default `Qwen/Qwen2.5-7B-Instruct`; cache `_pmi_cache/`; optional warm worker `pmi_worker.py`) |
| `_llm/llm_backend.py`, `_llm/serve_qwen.sh` | stdlib client for an OpenAI-compatible vLLM endpoint (`LLM_ENDPOINT`, `LLM_MODEL`) and the vLLM launch used for Qwen2.5-32B-Instruct (TP=4, port 8765, guided decoding) |

## Running

Set `SCISLOP_ROOT` and build the layout once (`bash scripts/make_root.sh`, then
`python3 scislopbench/build/materialize_bench.py`). Then, from the repository root:

```bash
# deterministic measures, no GPU
python3 benchmark/run_measure_release.py --half fars --checker xsec_ref
python3 benchmark/run_measure_release.py --half fars --checker macro_redund
python3 benchmark/run_measure_release.py --half fars --checker citation
python3 benchmark/run_measure_release.py --half fars --checker evidence_gap      # same with --half a4s

# Argument graph: labels need the Qwen2.5-32B server, PMI needs one GPU
bash scislop/_llm/serve_qwen.sh &                     # writes the endpoint; or export LLM_ENDPOINT=http://host:8765/v1
AG_LABEL_WORKERS=32 python3 benchmark/run_measure_release.py --half fars --checker argument_graph --extra "--stage labels --runs 3"
python3 benchmark/run_measure_release.py --half fars --checker argument_graph --extra "--stage pmi"

# Figure exposition: transcripts of the method figures are read from Artifacts/candidte/_census_0914/pairs165_full/transcripts*.jsonl
cd scislop/Artifacts/fig_exposition/code && python3 measure.py            # FARS half; add --transcribe to re-run the VLM on new crops
python3 benchmark/benchA4S/scripts/run_figexp_a4s.py [--transcribe]       # Agents4Science half
```

Each `code/` folder has a `RUN.md` with the exact invocations and a `PROMPT_*.txt` with the prompt sent to the model.
`prefetch_labels.py` and `cache_status.py` (Argument graph) fill and inspect the label cache; the ablations over
judge models (gemma-2-27b, Mistral-Small-24B, Qwen2.5-14B) and PMI models (GPT-J-6B, Mistral-7B, Qwen2.5-14B/1.5B) are
driven by `benchmark/bench165/scripts/run165_ag_judge.sh` and `run165_ag_pmi.sh` through `MODEL_TAG`, `LLM_MODEL`
and `PMI_MODEL`.

## Inputs of Figure exposition

`Artifacts/candidte/_census_0914/` holds the figure manifest of the FARS half (`pairs165_full/data/pairs165_manifest.json`,
which method figure of which paper, page and box), the hand-checked human-side figure positions
(`pairs165_hu_overrides.json`), the cached VLM transcripts (`pairs165_full/transcripts.jsonl`, `transcripts_fix.jsonl`)
and the two transcription scripts. The crops and the third-party PDFs they were cut from are not redistributed; the
crops are regenerated from the manifest with `pairs165_transcribe.py`. `leak_verdicts.py` holds the calibrated
decisions for the *experimental content* kind (Appendix A).

## Notes

- `Argument/citation/code/measure.py` computes the v4 isolation rate used in the paper; its module docstring and
  `CHECKER` string still carry the wording of the earlier v3 (vague-relation rate), which is kept behind `--claims`.
- Every model call is greedy, cached, and made to an open model served locally; no measurement calls a paid or closed
  model.
