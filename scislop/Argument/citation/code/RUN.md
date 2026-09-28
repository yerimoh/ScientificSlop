# citation — Run Guide (v4)

- **Model**: none for the score. The item is deterministic (citation keys + frozen cue lists). The optional claims layer (`--claims`) uses Qwen2.5-32B-Instruct bf16, greedy, via `_llm/llm_backend.py` (server: `sr 4 48 --qos=${SLURM_QOS} bash artifact-ai2science/_llm/serve_qwen.sh`; **never fp16, Qwen 32B degenerates**), prompt `PROMPT_relation.txt`, one call per paragraph with the preceding paragraph as context.
- **Input**: Introduction and Related Work paragraphs of the expanded tex, lightly cleaned so `\cite` keys survive.
- **Measurement**: per citing sentence the 0909 probe tags; weaving = {BUNDLE, MULTI, REL}; `slop_score = isolated citing sentences / citing sentences`, weak below 8 citing sentences. Paragraph observation `rw_chain_paragraph_rate` (>=3 works, no weaving sentence). The claims layer, when on, runs the v3 pipeline unchanged and reports `claims_vague_score` and its breakdowns as observations; it never enters the score.
- **Metric keys**: `slop_score, coverage, n_citing_sentences, n_weaving_sentences, co_citation_rate, tag_counts, rw_chain_paragraph_rate, self_positioning_rate, n_eligible_paragraphs`; with `--claims` additionally `claims_vague_score, claims_coverage, n_relation_claims, concrete_rate, vague_rate, inconclusive_rate, concrete_rate_by_kind, self_claim_concrete_rate, by_section, quote_unverified_rate, target_unverified_rate, no_claim_eligible_paragraph_rate`.

## Run
```bash
cd paper/draft_v6/slop/Argument/citation/code
python3 measure.py                          # full census, deterministic, no GPU, seconds -> ../results/
python3 measure.py --only FA0001            # single paper
python3 measure.py --claims --runs 3        # + LLM claims layer (evidence records; median of three runs)
```
Outputs in `../results/`: `papers.jsonl` (per paper), `paragraphs.jsonl` (per paragraph, per-sentence tags), `summary.json`; with `--claims` also `claims.jsonl` (every verified relation claim).

## Regression cases (must hold after any change)
| case | expected |
|---|---|
| "(cite A; cite B)" in one bracket | BUNDLE, sentence weaving |
| "A (cite A) does X, while B (cite B) does Y" | MULTI (+REL with a cue), weaving |
| "Unlike A (cite A), we ..." | REL1 only, sentence isolated (relation to self, reported apart) |
| "A did X (cite A). B did Y (cite B)." as two sentences | both isolated |
| paragraph citing >=3 works, no weaving sentence | counts into rw_chain_paragraph_rate |
| paper with < 8 citing sentences | slop_score present, weak = true |
| paraphrase preserving `\cite` keys and cue words | score unchanged |
| v3 archive | `measure.py.bak_0911v3` reproduces the old vague-rate score |
