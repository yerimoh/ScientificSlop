"""B3i reviewer backbone — CMU Paper Reviewer System, ported onto Qwen2.5-32B.

Source: github.com/prometheus-eval/cmu-paper-reviewer (open source).
Native pipeline: Mistral OCR -> OpenHands agent (GPT-5.5 / Claude-Opus, sampled
from config) with file-editor + terminal + task-tracker tools + Tavily MCP
literature search -> the agent may WRITE AND RUN verification code -> a markdown
review of at most `max_items` critical issues -> LaTeX PDF.

The review RUBRIC and OUTPUT CONTRACT are pulled from the repo's
backend/reviewer_prompt.py (verbatim where quoted below; connective wording
paraphrased). Defining feature of the review artifact = a small set of the MOST
critical issues, each written as Claim / Evidence(quotes) / Concrete Action Item,
with a citation list. (The agent's code-execution and web tools are its runtime
signature; see deviations.)

Deviations (logged for the write-up):
  * LLM = Qwen2.5-32B (repo rule, README.md 8.1) instead of the OpenHands agent
    backbone. Single-shot generation, not an agentic tool loop.
  * NO verification-code execution and NO Tavily literature search. Those are the
    agent's runtime tools; reproducing them needs an OpenHands sandbox and paid
    backends, out of scope for a review-generation baseline. Dropped and recorded
    (same spirit as B3a dropping the PDF path / GPT-4o). The Claim/Evidence/Action
    STRUCTURE and the max-5-critical-issues rubric are what we keep.
  * Input = LaTeX source (what the survival loop edits), not an OCR'd PDF.
  * Default preset = NeurIPS (FARS corpus is ML papers). Env B3I_PRESET=nature.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
from llm_backend import llm  # noqa: E402

MAX_ITEMS = int(os.environ.get("B3I_MAX_ITEMS", "5"))     # repo default max_items=5
PRESET = os.environ.get("B3I_PRESET", "neurips").lower()
TRUNC_LADDER = [18000, 12000, 8000]

# ---- rubric presets (from backend/reviewer_prompt.py) ------------------------
CRITERIA_NEURIPS = """Evaluation criteria (in order of importance):
1. Originality: Are the tasks or methods new?
2. Quality: Is the submission technically sound?
3. Clarity: Is the submission clearly written?
4. Significance: Are the results important?"""

CRITERIA_NATURE = """Evaluation criteria (in order of importance):
1. Validity: Does the manuscript have significant flaws which should prohibit its publication?
2. Conclusions: Are the conclusions and data interpretation robust, valid and reliable?
3. Originality and significance.
4. Data and methodology.
5. Appropriate use of statistics and treatment of uncertainties.
6. Clarity and context."""

CRITERIA = CRITERIA_NATURE if PRESET == "nature" else CRITERIA_NEURIPS

# Core instruction + item contract (verbatim / near-verbatim from reviewer_prompt.py)
SYSTEM = (
    "You are an expert reviewer for a top scientific venue. "
    f"Your task is to write a review in markdown format, where your review must "
    f"contain at most {MAX_ITEMS} items (from most significant to least significant).\n\n"
    + CRITERIA +
    "\n\nEach review item MUST use this structure:\n"
    "### Claim\n<a single critical issue stated plainly>\n"
    "### Evidence\n<quote the relevant text from the paper, then a comment on each "
    "quote explaining why it is a problem>\n"
    "### Concrete Action Item\n<exactly one of: \"Fix the writing\" or "
    "\"Add new implementation\">, followed by the specific change the authors should make.\n\n"
    "End the review with a Citation List of any references you invoked, with titles.\n"
    "Only point out problems that meaningfully affect the paper based on the "
    "evaluation criteria. Producing fewer items than the cap is appropriate only "
    "when you have genuinely exhausted the significant issues.")


def perform_review(paper_text, log=None):
    """Returns (review_dict | None, info). review_dict = {'review_markdown', 'n_items'}."""
    for level, pp in enumerate(TRUNC_LADDER):
        prompt = ("Here is the paper to review:\n\n```\n" + paper_text[:pp]
                  + "\n```\n\nWrite the review now, following the required structure "
                    f"and the at-most-{MAX_ITEMS}-items rule.")
        md = llm(prompt, system=SYSTEM, max_tokens=3000)
        if md and md.strip() and "### Claim" in md:
            n_items = md.count("### Claim")
            review = {"review_markdown": md.strip(), "n_items": n_items,
                      "preset": PRESET, "Decision": None, "Overall": None}
            info = {"level": level, "n_items": n_items}
            if log:
                log.write_text(json.dumps({"info": info, "final": review}, indent=2))
            return review, info
    return None, {"level": None, "n_items": 0}


if __name__ == "__main__":
    p = Path(sys.argv[1])
    if p.is_dir():
        text = "\n".join(f.read_text(errors="ignore")
                         for f in [p / "main.tex"] + sorted((p / "sections").glob("*.tex"))
                         if f.exists())
    else:
        text = p.read_text(errors="ignore")
    rev, info = perform_review(text)
    print((rev or {}).get("review_markdown", "")[:2000]); print(info)
