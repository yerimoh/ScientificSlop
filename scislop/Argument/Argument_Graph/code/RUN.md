# Argument_Graph: Run Guide (v3)

- **Models**: two, neither judges. Labels by Qwen2.5-32B-Instruct bf16, greedy, via `_llm/llm_backend.py` (server: `sr 4 48 --qos=${SLURM_QOS} bash artifact-ai2science/_llm/serve_qwen.sh`; never fp16), prompt `PROMPT_label.txt`, one call per sentence with the numbered introduction as context, 3 runs, majority. Pairwise scoring by Qwen2.5-7B-Instruct bf16 as a local causal LM (`_common/lm_score.py`, weights from the HuggingFace cache, `HF_HUB_OFFLINE=1`), one GPU; it returns log-probabilities and is never prompted.
- **Input**: Introduction of the prose view, sentences by the shared splitter, at least 4 alphabetic words each.
- **Measurement**: every sentence labelled superiority / prior_limitation / design_choice / none; `pmi(j -> i)` for every ordered pair with the two sentences concatenated by the code; `pred(i)` = strongest predictor; a key claim is argued when `pred(i)` precedes it and declared otherwise; `build_up(i)` = forward predictor chain length; `ABU` = mean build-up over key claims minus the same under 300 random sentence orders. `slop_score = declared / key claims`, weak below 3 key claims.
- **Metric keys**: `slop_score, coverage, n_key_claims, n_sentences, argued_rate, argued_expected, build_up_mean, build_up_null, ABU, ABU_z, ABU_null_pct, deep_rate, key_by_kind, slop_by_kind, build_up_by_kind, ABU_k2, ABU_k3, argued_rate_k2, argued_rate_k3, label_unanimity, predictor_adjacent_rate, predictor_distance_median`.

## Run
The two stages need different hardware and are cached separately, so the usual order is labels first with the server up, then PMI with one GPU. `--stage all` runs whichever stage is not yet cached.

```bash
cd paper/draft_v6/slop/Argument/Argument_Graph/code
python3 measure.py --stage labels --runs 3                 # server up: label every sentence, cache (about 1 s per sentence at 8-way)
python3 measure.py --stage pmi                              # one GPU: PMI matrices + graph + null, from cached labels (about 10 s per paper)
python3 measure.py --only FA0027,2402.14016                 # one pair, both stages from cache
sr 1 48 --qos=${SLURM_QOS} python3 measure.py --stage pmi   # the PMI stage on the cluster
```
Outputs in `../results/`: `papers.jsonl` (per paper), `claims.jsonl` (one evidence record per key claim: claim sentence and span, strongest predictor and span, before/after, build-up), `summary.json` (AI vs HU on score and `ABU`, key-count bands, per kind, k-sensitivity, unanimity list, within-corpus confounds). PMI matrices persist in `_common/_pmi_cache/`; label calls in `_common/_llm_cache/`.

A paper whose labels are not cached while the server is down is written with `status: labels_pending`; one whose PMI fails is `pmi_pending`. Neither enters group statistics.

## Regression cases (must hold after any change)
| case | expected |
|---|---|
| "We propose X" / "Our contributions are" / "as shown in Table 2" | label `none` (announcement, contribution, pointer) |
| "Existing methods like Y fail to Z" | label `prior_limitation`, a key claim |
| key claim whose strongest predictor is the next sentence | declared; build_up 0; counts toward slop_score |
| key claim whose strongest predictor is two sentences earlier, itself predicted from earlier | argued; build_up >= 2 |
| the same introduction with sentences shuffled | identical PMI matrix up to permutation (scorer never saw positions); ABU differs |
| introduction with 2 key claims | slop_score present, weak = true |
| introduction with 0 key claims | slop_score N/A, ABU None, coverage 0 |
| server down, labels not cached | status labels_pending, no zeros written |
| `PMI_MODEL` changed | every matrix recomputed (cache keyed on model) |
| v2 archive | `measure.py.bak_0910v2` reproduces the old supported-key-claim rate |
