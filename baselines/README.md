# Baselines: token-level detectors and automated reviewers

The comparison systems of Table 2 (§5.1). Detectors read the prose view (`body.txt`); reviewers read the full source.
Runners that apply them to the benchmark are in [`../benchmark/`](../benchmark/README.md#runners); the ICLR corpora
use the same modules through `../iclr_analysis/`.

## Detectors (`detectors/`)

| Script | Method | Setting used for Table 2 | Direction |
|---|---|---|---|
| `binoculars.py` | Binoculars (observer/performer perplexity ratio) | `tiiuae/falcon-7b` and `falcon-7b-instruct`; inputs beyond the 2048-token context are scored in non-overlapping windows and averaged (`run165_bino_faithful.sh`). The script's default (Qwen2.5-7B, 6 × 512-token windows) is the reduced setting used for exploration | lower = more AI-like |
| `detectgpt.py` | DetectGPT (perturbation curvature) | scorer `gpt-j-6b`, perturber `t5-3b`, 100 perturbations (`run165_dgpt_faithful.sh`); default reduced setting Qwen2.5-1.5B / t5-large / 20 perturbations | higher = more AI-like |
| `fast_detectgpt.py` | Fast-DetectGPT (sampling discrepancy) | sampling model `gpt-j-6b`, scoring model `gpt-neo-2.7B` | higher = more AI-like |
| `nts.py` | NTS curvature score | `falcon-7b`, first 512 tokens | higher = more AI-like |
| `extract_texts.py`, `analyze_detectors.py` | helpers from the earlier ICLR head-to-head (text extraction, per-year summaries) | | |

Each script reads a JSONL of `{id, text}` (`texts.jsonl`, written by the view builders) and writes one JSON line per
paper with the score and window counts. Pangram (Table 2 of the paper does not include it; it appears in Fig. 4 and
in pairing condition H1) is called by `../benchmark/benchA4S/scripts/pangram_bench.py`.

## Automated reviewers (`reviewers/`)

| Folder | System | Model | Output used |
|---|---|---|---|
| `cyclereviewer/` | CycleReviewer | `WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B` on vLLM (`serve_cyclerev.sbatch`), temperature 0.4 | `final.Overall` |
| `ai_scientist/` | The AI Scientist reviewer (`perform_review` port) | Qwen2.5-32B-Instruct served as `qwen`; ensemble of 5 reviews at temperature 0.75 plus an area-chair meta-review | `final.Overall` |
| `cmu_reviewer/` | CMU paper-reviewer rubric | Qwen2.5-32B-Instruct | gives no overall score; not in the tables |

`cyclerev_reviewer.py`, `sakana_reviewer.py` and `cmu_reviewer.py` expose one function each that takes the paper text
and returns the parsed review JSON; `b3i_run.py` and the `run_b*.sh` / `*.sbatch` files are the batch drivers used on
the earlier 165-pair runs. On the benchmark they are invoked through `bench_reviews.py` (FARS half) and
`bench_reviewsA4S.py` (Agents4Science half). Servers are addressed through `LLM_ENDPOINT` (Qwen) and the endpoint file
written by `serve_cyclerev.sbatch`.
