# SciSlopHarness and the revision baselines

Code for §4 (SciSlopHarness), Appendix C (implementation details), Appendix D (revision baselines), the component
ablation (Fig. 6) and the figures and tables of §5.3–5.4.

```
harness/
  harness.py            the round loop: locate slop instances, stage SLOP_FINDINGS.md + SciSlop.md, call the editor,
                        apply edits, run the hard preservation guards, measure, hand the round to the reviewer gate
  quality_gate.py       the reviewer gate: reviews proposed changes three at a time against the manuscript and the
                        experiment records, returns KEEP / REVERT per change, and the retirement judgement
  gate_settings.json    gate prompts and thresholds
skill/SciSlop_v0.5.md   the skill file staged as SciSlop.md: definitions of the six patterns, repair directions
                        (Table 10), preservation checks (Table 11)
drivers/                batch runs, resumption, reports and figures (run_batch_v14.py is the run behind Fig. 5)
revision_baselines/     the four general/slop-aware revision arms of Appendix D (run_arm.py, prompts.py)
ablation/               the component ablation of Fig. 6 (run_ablation.py, summarize_ablation.py, fig_component_*.py)
```

## Configuration used in the paper (Table 9)

The configuration is the `haiku_sonnetgate` entry of `drivers/run_final1.py` together with `FAST` in `drivers/run_v2.py`,
launched by `drivers/run_batch_v14.py`. Everything is passed to `harness.py` as environment variables:

| Setting | Value | Variable |
|---|---|---|
| Editor | `claude-haiku-4-5-20251001`, per-file `<<<EDIT>>>` blocks with the other files as context, 1024 thinking tokens | `SH_EDITOR_MODE=edits_parallel`, `SH_EDITOR_THINK`, `SH_EDITOR_MAX_TURNS` |
| Reviewer gate | `claude-sonnet-5`, no tools, text only, changes reviewed three at a time, 2048 thinking tokens | `SH_GATE=1`, `SH_GATE_TEXT=1`, `SH_GATE_CHUNK=3`, `SH_GATE_INLINE=1`, `SH_GATE_THINK` |
| Retirement call | `claude-haiku-4-5-20251001` | `SH_GATE_RETIRE_SPLIT=1`, `SH_RETIRE_MODEL` |
| Rounds | at most 3 | `SH_ROUNDS=3` |
| Skill | `SciSlop_v0.5.md`, up to 8 located instances per pattern | `SH_SKILL`, `SH_LOCATION_CAP=8` |
| Patterns fed back | the four deterministic measures plus Argument graph (labels from the Qwen server, PMI from a warm worker) and Figure exposition (bounding boxes from a VLM, fill by LaMa inpainting) | `SH_EXTRA_ITEMS=1`, `AG_PMI_DAEMON=1`, `SH_FIG_METHOD=lama`, `SH_FIG_READER=claude` |
| Evidence gap | an acknowledged gap scores 0.5 | `SH_EG_ACK=1` |
| Experiment records | `code/exp/EXPERIMENT_RESULTS/**` of the FARS paper, copied into `materials/` (no `.tex/.pdf/.png/.pkl/.pt`, ≤ 400 KB per file, ≤ 60 files) | `revision_baselines/common.py::stage_materials` |

The editor and the reviewer are called through the Claude Code CLI (`claude -p ... --output-format json`).
`SH_API_BASE` and `SH_API_KEY_FILE` point the CLI at a gateway; without them it uses the CLI's own login.
The run stops when no instance remains, the editor makes no change, the gate reverts a whole round twice, or after
round 3 (Appendix C.4).

```bash
export SCISLOP_ROOT=...            # with fars/papers/<code>_*/ present (LaTeX + EXPERIMENT_RESULTS)
export LLM_ENDPOINT=http://host:8765/v1        # Qwen2.5-32B server for Argument-graph labels
python3 scislopharness/drivers/run_batch_v14.py --papers FA0005,FA0023 --workers 4 --suffix _v14
```

Per paper the run writes `R1..R3/` manuscript trees, `logs/` (prompts, responses, SLOP_FINDINGS, patches),
`gate/R<n>/{CHANGES.md,VERDICTS.json,RETIRE.json}`, `measure/R<n>/summary.json`, `trajectory.json`, `figures/`,
`ROUNDS.md` and `CASE_PACK.md`; per batch `results.jsonl`, `batch.log` and `batch_summary.json`.

## Revision baselines (Appendix D)

`revision_baselines/run_arm.py <arm> <paper>` runs one arm for three rounds with the same editor model
(`EOR_MODEL`, default `claude-haiku-4-5-20251001`) and the shared protocol of Appendix D.1 (same staging, same guards,
same measurement by `measure_tree.py`):

| Arm | Paper name | What the editor sees |
|---|---|---|
| `a1_base` | Base prompting | the manuscript and a generic revision request, no tools |
| `a2_code` | Claude Code | the Claude Code agent in restricted mode with its stock system prompt |
| `a3_review` | Reviewer-based refinement | a review from Qwen2.5-32B-Instruct (`review_qwen.py`, input truncated at 50k characters, fallback 24k / 14k) and the request to address it |
| `a4_slop` | Slop-aware revision | the pattern definitions plus up to 12 located instances per pattern (Table 13), no reviewer gate |

`prompts.py` holds every prompt verbatim; `measure_rounds.py` and `aggregate.py` score the rounds and pool them;
`fig_retention.py` and `make_paper_figures.py` draw the appendix figures.

## Figures and tables of §5.3–5.4

| Item | Script | Reads |
|---|---|---|
| Fig. 5 (slop minus human mean, rounds 0–3, six measures) and its data file | `drivers/fig_harness_result.py` | the harness batches and the four baseline arms; human means from the benchmark |
| Table 12 (sum of absolute distances after round 3; the 63 % figure is 1 − ours/best baseline) | `drivers/tab_harness_gap.py` | the data file written by `fig_harness_result.py` |
| Table 4 and Table 14 (matched revision cases) | `drivers/case_study_v2.py`, `casepick.py`, `case_study_extract.py` | run trees |
| Per-round and cost reports | `drivers/rounds_table.py`, `cost_report.py`, `summarize*.py`, `final*_report.py`, `compare_versions.py` | run trees |
| Figure edits (bounding boxes, inpainting, comparison grids) | `drivers/figure_edit.py`, `figure_bands.py`, `figure_inpaint.py`, `figure_grid.py`, `figure_compare.py` | figures of a run |
| Specimen search (is there a verbatim example in the records?) | `drivers/specimen_search.py`, `specimen_t2.py` | experiment records |

## Component ablation (Fig. 6)

`ablation/code/run_ablation.py --arm {nogate,gateonly,noloc} --papers ... --rounds 3` reruns the harness with one
component removed: `nogate` keeps definitions and located instances but drops the reviewer gate; `gateonly` keeps the
gate with a generic editor prompt; `noloc` keeps definitions and the gate but gives no locations. The full system is
the `run_batch_v14` run. `ablation/skills/` holds the reduced skill files of the earlier wording arms (name / +def /
+def+loc), `summarize_ablation.py` pools the arms, and `fig_component_ratio.py`, `fig_component_tradeoff.py` and
`fig_component_summary.py` draw the figure (distance to the human mean against guard breaks). `pmi_qmid.sbatch` is the
SLURM job for the PMI worker the ablation shares.
