"""Faithful port of the Sakana AI-Scientist reviewer (perform_review) onto the
repo's Qwen2.5-32B vLLM backend.

Source: github.com/SakanaAI/AI-Scientist ai_scientist/perform_review.py
(arXiv 2408.06292). A verbatim copy is kept next to this file as
_reference_sakana_perform_review.py. All prompts below (system, NeurIPS form,
template instructions, reflection, meta-reviewer) are VERBATIM from that file.

Pipeline (canonical launch_scientist.py settings in parentheses):
  1. base prompt = NeurIPS review form + few-shot example(s) + paper text
  2. ensemble of N independent reviews, temperature 0.75  (N=5)
  3. AC meta-review aggregates them; numeric fields replaced by ensemble means
  4. self-reflection rounds on the aggregated review          (5, temp 0.1)

Deviations from the original, all forced by the local serving stack and logged
here so the baseline write-up can cite them:
  * LLM = Qwen2.5-32B-Instruct (repo-wide decision, REVIEWER_SYSTEMS_SURVEY.md:
    "the reviewer LLM is fixed to Qwen2.5-32B") instead of GPT-4o.
  * Input is the LaTeX source (what the survival loop edits), not a PDF dump.
  * 32k serving context => adaptive truncation ladder over (few-shot, paper)
    lengths, mirroring the 50k->24k->14k ladder already used by b3_run.py.
  * Sakana's code sends reviewer_reflection_prompt with its {current_round}
    placeholders unformatted; we format them (the evident intent).
  * Ensemble / reflection counts default to 3 / 2 for GPU budget; override with
    B3A_ENSEMBLE / B3A_REFLECT env vars (5 / 5 = canonical Sakana).
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
from llm_backend import endpoint  # noqa: E402

HERE = Path(__file__).resolve().parent
NUM_ENSEMBLE = int(os.environ.get("B3A_ENSEMBLE", "3"))
NUM_REFLECT = int(os.environ.get("B3A_REFLECT", "2"))   # Sakana num_reflections (2 => 1 reflection pass)
TEMP_ENSEMBLE = 0.75    # hardcoded in perform_review's ensemble branch
TEMP_BASE = 0.1         # launch_scientist.py review temperature
MAX_TOKENS = int(os.environ.get("B3A_MAX_TOKENS", "3000"))   # Sakana default 3000; raised only for the papers whose
# review JSON was cut off at the limit and therefore failed to parse (see review/scripts/retry_b3a_failed.py)
TIMEOUT = 600

# (few-shot chars, paper chars) ladder — largest first. Empirical 2026-07-21:
# the served Qwen2.5-32B degenerates into token soup once the TOTAL prompt
# exceeds ~32-35k chars (well below the 32k-token window; same failure b3 hit,
# fixed there with its 50k->24k->14k ladder). Form ≈ 8k chars and reflection
# history adds ~4k, so keep base prompt ≤ ~27k chars.
TRUNC_LADDER = [(5000, 14000), (3000, 10000), (2000, 7000)]

# ---- verbatim Sakana prompts -------------------------------------------------
reviewer_system_prompt_base = (
    "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue."
    "Be critical and cautious in your decision."
)

reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + "If a paper is bad or you are unsure, give it bad scores and reject it."
)

template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""

neurips_form = (
    """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.

1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited
  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work
  - Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?

2. Questions: Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.

3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If not, please include constructive suggestions for improvement.
In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.

4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.

5. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
  4: excellent
  3: good
  2: fair
  1: poor

6. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
  4: excellent
  3: good
  2: fair
  1: poor

7. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
  4: excellent
  3: good
  2: fair
  1: poor

8. Overall: Please provide an "overall score" for this submission. Choices:
  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations

9. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
  5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
  4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
  3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
"""
    + template_instructions
)

reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
Include any other factors that you think are important in evaluating the paper.
Ensure the review is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your review.
Stick to the spirit of the original review unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

meta_reviewer_system_prompt = """You are an Area Chair at a machine learning conference.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""

SCORE_LIMITS = [
    ("Originality", (1, 4)), ("Quality", (1, 4)), ("Clarity", (1, 4)),
    ("Significance", (1, 4)), ("Soundness", (1, 4)), ("Presentation", (1, 4)),
    ("Contribution", (1, 4)), ("Overall", (1, 10)), ("Confidence", (1, 5)),
]

# ---- Qwen chat with message history (llm_backend.llm is single-turn) ---------
def chat(messages, temperature, max_tokens=MAX_TOKENS, timeout=TIMEOUT):
    body = json.dumps({"model": "qwen", "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    try:
        req = urllib.request.Request(f"{endpoint()}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return r["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def extract_json(txt):
    """Sakana's extract_json_between_markers equivalent: balanced-brace scan."""
    if not txt:
        return None
    txt = txt.replace("```json", "").replace("```", "")
    start = txt.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(txt)):
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:i + 1])
                except Exception:
                    return None
    return None


# ---- few-shot examples (Sakana repo files, num_fs_examples=1 canonical) ------
def fewshot_prompt(max_chars, num_fs_examples=1):
    prompt = """
Below are some sample reviews, copied from previous machine learning conferences.
Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.
"""
    names = ["132_automated_relational", "attention", "2_carpe_diem"][:num_fs_examples]
    per = max_chars // max(1, len(names))
    for n in names:
        paper_text = (HERE / "fewshot_examples" / f"{n}.txt").read_text(errors="ignore")[:per]
        review_text = json.loads((HERE / "fewshot_examples" / f"{n}.json").read_text())["review"]
        prompt += f"""
Paper:

```
{paper_text}
```

Review:

```
{review_text}
```
"""
    return prompt


def base_prompt(paper_text, fs_chars, paper_chars):
    return (neurips_form + fewshot_prompt(fs_chars)
            + f"""
Here is the paper you are asked to review:
```
{paper_text[:paper_chars]}
```""")


def valid_review(o):
    if not isinstance(o, dict):
        return False
    if not (isinstance(o.get("Summary"), str) and o["Summary"].strip()):
        return False
    if not (o.get("Strengths") and o.get("Weaknesses")):
        return False
    ov = o.get("Overall")
    return isinstance(ov, (int, float)) and 1 <= ov <= 10


# ---- main entry: perform_review port ------------------------------------------
def perform_review(paper_text, num_ensemble=NUM_ENSEMBLE, num_reflections=NUM_REFLECT,
                   log=None):
    """Returns (review_dict | None, info) — info = {'level': .., 'n_valid': ..}."""
    for level, (fs_c, pp_c) in enumerate(TRUNC_LADDER):
        bp = base_prompt(paper_text, fs_c, pp_c)
        user0 = {"role": "user", "content": bp}

        # (2) ensemble of independent reviews, temp 0.75
        parsed = []
        raw = []
        for _ in range(num_ensemble):
            out = chat([{"role": "system", "content": reviewer_system_prompt_neg}, user0],
                       temperature=TEMP_ENSEMBLE)
            raw.append(out)
            o = extract_json(out)
            if valid_review(o):
                parsed.append(o)
        if not parsed:
            continue  # context overflow / degenerate output -> shorter ladder step

        # (3) AC meta-review aggregates the ensemble
        review = None
        if len(parsed) > 1:
            review_text = ""
            for i, r in enumerate(parsed):
                review_text += f"""
Review {i + 1}/{len(parsed)}:
```
{json.dumps(r)}
```
"""
            meta_out = chat(
                [{"role": "system",
                  "content": meta_reviewer_system_prompt.format(reviewer_count=len(parsed))},
                 {"role": "user", "content": neurips_form + review_text}],
                temperature=TEMP_BASE)
            meta = extract_json(meta_out)
            if valid_review(meta):
                review = meta
        if review is None:
            review = parsed[0]

        # numeric fields -> rounded ensemble means (Sakana verbatim behaviour)
        for score, limits in SCORE_LIMITS:
            vals = [r[score] for r in parsed
                    if isinstance(r.get(score), (int, float)) and limits[0] <= r[score] <= limits[1]]
            if vals:
                review[score] = int(round(sum(vals) / len(vals)))

        # (4) self-reflection on the aggregated review, temp 0.1
        msgs = [{"role": "system", "content": reviewer_system_prompt_neg}, user0,
                {"role": "assistant", "content": f"""
THOUGHT:
I will start by aggregating the opinions of {num_ensemble} reviewers that I previously obtained.

REVIEW JSON:
```json
{json.dumps(review)}
```
"""}]
        for j in range(num_reflections - 1):
            msgs.append({"role": "user", "content": reviewer_reflection_prompt.format(
                current_round=j + 2, num_reflections=num_reflections)})
            out = chat(msgs, temperature=TEMP_BASE)
            msgs.append({"role": "assistant", "content": out})
            o = extract_json(out)
            if valid_review(o):
                review = o
            if "I am done" in out:
                break

        if log:
            log.write_text(json.dumps({"level": level, "n_valid": len(parsed),
                                       "ensemble": parsed, "final": review}, indent=2))
        return review, {"level": level, "n_valid": len(parsed)}
    return None, {"level": None, "n_valid": 0}


if __name__ == "__main__":
    # smoke: python3 sakana_reviewer.py <tex_file_or_dir>
    p = Path(sys.argv[1])
    if p.is_dir():
        text = "\n".join(f.read_text(errors="ignore")
                         for f in [p / "main.tex"] + sorted((p / "sections").glob("*.tex"))
                         if f.exists())
    else:
        text = p.read_text(errors="ignore")
    rev, info = perform_review(text)
    print(json.dumps(rev, indent=2)[:2000])
    print(info)
