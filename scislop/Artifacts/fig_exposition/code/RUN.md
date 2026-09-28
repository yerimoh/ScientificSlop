# fig_exposition — Run Guide
- **Model**: Qwen2.5-VL-32B-Instruct, one whole-figure transcription call per figure (`PROMPT_transcribe.txt`), the answer kept in full. Nothing else calls a model. The six element patterns are deterministic and the experimental-content kind is hand-verified in `leak_verdicts.py`. Every call is cached, so a rerun needs no GPU.
- **Input**: the method figure of each side of a benchmark pair. AI = the generation pipeline's `framework_overview` image. Human = the figure named by the hand localisation table `../../candidte/_census_0914/pairs165_hu_overrides.json`, cut from the rendered PDF page. A human paper marked `none` there draws no method figure and is left out of the run, never scored as clean.
- **Metric keys**: per figure `kinds, n_kinds, words, slop_score (kinds internalised / 6), coverage`; per element `verdict` in {internalised, absent} with `decided_by` in {pattern, hand}.

## Run
```bash
cd paper/draft_v6/slop/Artifacts/fig_exposition/code
python3 measure.py                      # both corpora from the cache, no GPU
python3 measure.py --corpus AI          # FARS only
python3 measure.py --only FA0006,FA0209 # a named subset
python3 measure.py --transcribe         # read figures missing from the cache (GPU)
```
Outputs in `../results/`: `elements.jsonl`, `papers.jsonl`, `summary.json`.

The transcription pass wants a GPU. Submit it the usual way and the cache makes every later run free:
```bash
sr 4 48 --qos=${SLURM_QOS}           # then python3 measure.py --transcribe
```

## Reading rules
- **Report the per-kind table beside the score.** A single count hides which kind carries it, and one kind (`enumerated_stage`) separates the corpora hardly at all. Report the leave-one-kind-out effect with it.
- **Report the human rate beside every AI rate.** Human authors do number stages and do tag their method; the item is a difference of degree on five kinds and of kind on none.
- **Never narrow a pattern by removing a human hit.** A pattern that fires on a technical term is rewritten and the whole census re-run. The reinforcement-learning term "Advantage" fired `thesis_box` twice on the human side and the pattern was narrowed to headed phrases.
- **Text mass stays out of the score.** Words per figure separates the corpora on the pairs and reverses on another corpus, and the human figure is a crop that can lose text at its border.
- **A paper with no method figure is NA.** It is not listed by `figures()` and must never enter a denominator.
- **The hand-decided kind needs a second reader** before its rate is quoted; `leak_verdicts.py` carries one reader's verdicts with a one-line reason each.

## Read before quoting a number
`../../candidte/_census_0914/README.md`, sections "fig_slide" and "fig_slide v2" (the census kept the working name). It carries the per-kind counts, the count-score AUROC with its bootstrap interval, the leave-one-kind-out table, every human hit with the line that matched, and the held-out run. The held-out run is the one to read first. Patterns chosen on these pairs carry to unseen FARS figures against an unseen human set, and they do not separate a second generation system from humans, so the item is reported as a signature of one figure pipeline and not as a property of AI authorship.
