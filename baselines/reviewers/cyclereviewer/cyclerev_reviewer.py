"""Faithful port of the CycleReviewer reviewer (WestlakeNLP) onto a locally
served vLLM endpoint.

Source: github.com/zhu-minjun/Researcher ai_researcher/cycle_reviewer.py +
ai_researcher/utils.py (arXiv 2411.00816, ICLR 2025 "CycleResearcher: Improving
Automated Research via Automated Review"). A verbatim snapshot is kept next to
this file as _reference_cycle_reviewer.py.

Unlike B3a (prompt pipeline re-hosted on Qwen), CycleReviewer's defining feature
IS the fine-tuned open-source model (Llama-3.1 SFT + iterative preference RL on
ICLR reviews), so this baseline serves the ACTUAL released checkpoint:
    WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B   (70B/123B exist; 8B fits 1 GPU)
via vLLM OpenAI API (serve_cyclerev.sh -> cyclerev_endpoint.txt).

Pipeline (original evaluate(), all canonical):
  1. system prompt (verbatim below) + user = raw paper LaTeX
  2. single generation, temperature 0.4, top_p 0.95, max_tokens 7000,
     max_model_len 50000 (set server-side)
  3. the fine-tune emits 4 simulated ICLR reviewers (Summary / Soundness /
     Presentation / Contribution / Strengths / Weaknesses / Questions /
     Rating 1-10 / Confidence) + Meta Review + Paper Decision in one pass
  4. parsed by a verbatim port of get_reviewer_score (7B + 123B formats)

Deviations from the original, logged for the baseline write-up:
  * Serving: vLLM offline LLM class -> OpenAI-compatible server (same engine,
    same sampling params), so the survival loop's CPU shards can share one GPU.
  * Input is the LaTeX source tree text (what the survival loop edits) — same
    as the original tutorials, which feed paper['latex'].
  * Adaptive truncation ladder over paper chars (20k->14k->9k->6k) — the
    served 8B collapses into repetition loops (never reaching Paper Decision)
    beyond ~20-25k paper chars (see TRUNC_LADDER note); plus 1 resample per
    level when the output fails to parse (temp 0.4).
  * Original parser does float(rating_str[0]) — a '10' parses as 1.0. Fixed
    with a leading-number regex. The 7B parser also gates Summary extraction
    on '#@ Summary' (evident typo -> summary always ''); fixed to '## Summary'.
    Everything else is behaviour-faithful, including the 7B-format ->
    123B-format fallback dispatcher.
"""
from __future__ import annotations
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINT_FILE = HERE / "cyclerev_endpoint.txt"

TEMP = 0.4          # original SamplingParams
TOP_P = 0.95
MAX_TOKENS = 7000
TIMEOUT = 900
# Paper-chars ladder — largest first. Empirical 2026-07-23 (FA0001, served
# 8B): generation collapses into '### Soundness' repetition loops (no EOS, no
# Paper Decision) once the paper part exceeds ~20-25k chars — 25.3k = 0/8
# parses, 20k = 4/4, 15k = 4/4. Same cliff-style long-prompt collapse b3/B3a
# hit on the served Qwen (~32-35k chars there); same mitigation (B3a reviews
# papers at <=14k chars, so 20k here still sees more of the paper).
# repetition_penalty 1.05/1.1 does NOT rescue it (1/4, 0/4).
TRUNC_LADDER = [20000, 14000, 9000, 6000]
RESAMPLE_PER_LEVEL = 2                          # attempts per ladder level

# ---- verbatim CycleReviewer system prompt (whitespace preserved) --------------
SYSTEM_PROMPT = (
    "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:\n"
    "    \n"
    "            1. Summary: Briefly outline main points and objectives.\n"
    "            2. Soundness: Assess methodology and logical consistency.\n"
    "            3. Presentation: Evaluate clarity, organization, and visual aids.\n"
    "            4. Contribution: Analyze significance and novelty in the field.\n"
    "            5. Strengths: Identify the paper's strongest aspects.\n"
    "            6. Weaknesses: Point out areas for improvement.\n"
    "            7. Questions: Pose questions for the authors.\n"
    "            8. Rating: Score 1-10, justify your rating.\n"
    "            9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).\n"
    "    \n"
    "            Maintain objectivity and provide specific examples from the paper to support your evaluation.\n"
    "    \n"
    "            You need to fill out **4** review opinions."
)


def endpoint():
    return ENDPOINT_FILE.read_text().strip()


def chat(paper_text, temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS,
         timeout=TIMEOUT):
    """One CycleReviewer generation. Returns (text, err) — err='ctx' on
    context-length 400s (caller should drop a ladder level), 'other' else."""
    body = json.dumps({
        "model": "cyclereviewer",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": paper_text}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
    }).encode()
    try:
        req = urllib.request.Request(f"{endpoint()}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return r["choices"][0]["message"]["content"] or "", None
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")[:400]
        return "", ("ctx" if ("maximum context length" in msg or e.code == 400)
                    else f"http{e.code}")
    except Exception as e:
        return "", f"other:{str(e)[:80]}"


# ---- verbatim-port parsers (see module docstring for the one rating fix) ------
def _rating_num(s):
    """Original: float(s[0]) — breaks on '10'. Parse the leading number."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0


def get_reviewer_score_7B(generated_text):
    try:
        pred = {}
        reviews, review_rate, rating = [], [], []
        summary, soundness, presentation, contribution = [], [], [], []
        strengths, weaknesses, questions = [], [], []
        flag_for_ethics_review, confidence = [], []
        paper_decision, meta_review = '', ''
        for review in generated_text.split('**********\n'):
            if review != '':
                if '## Paper Decision\n\n' in review:
                    review, paper_decision = review.split('## Paper Decision\n\n')[:2]
                    paper_decision = paper_decision.split('\n')[0]
                    paper_decision = 'Accept' if 'accept' in paper_decision.lower() else 'Reject'
                    break
                if '## Meta Review\n\n' in review:
                    review, meta_review = review.split('## Meta Review\n\n')[:2]
                elif ('## Summary\n\n' in review and '## Soundness\n\n' in review
                        and '## Presentation\n\n' in review):
                    reviews.append(review)

                    def sec(tag, splitter='##'):
                        if f'## {tag}\n\n' in review:
                            return review.split(f'## {tag}\n\n')[1].split(splitter)[0]
                        return ''
                    summary.append(sec('Summary'))
                    soundness.append(sec('Soundness'))
                    presentation.append(sec('Presentation'))
                    contribution.append(sec('Contribution'))
                    strengths.append(sec('Strengths'))
                    weaknesses.append(sec('Weaknesses'))
                    questions.append(sec('Questions'))
                    flag_for_ethics_review.append(sec('Flag For Ethics Review'))
                    r = sec('Rating')
                    review_rate.append(r)
                    rating.append(_rating_num(r) if r else 0)
                    confidence.append(sec('Confidence', '******'))
        if paper_decision == '':
            return None
        pred['content'] = generated_text
        pred['reviews'] = reviews
        pred['summary'] = summary
        pred['review_rate'] = review_rate
        pred['rating'] = rating
        pred['soundness'] = soundness
        pred['presentation'] = presentation
        pred['contribution'] = contribution
        pred['strength'] = strengths
        pred['weaknesses'] = weaknesses
        pred['questions'] = questions
        pred['flag_for_ethics_review'] = flag_for_ethics_review
        pred['confidence'] = confidence
        pred['paper_decision'] = paper_decision
        pred['meta_review'] = meta_review
        pred['avg_rating'] = sum(rating) / len(rating)
        return pred
    except Exception:
        return None


def get_reviewer_score_123B(generated_text):
    try:
        pred = {}
        reviews, review_rate, rating = [], [], []
        summary, soundness, presentation, contribution = [], [], [], []
        strengths, weaknesses, questions = [], [], []
        flag_for_ethics_review, confidence = [], []
        paper_decision, meta_review = '', ''
        for review in generated_text.split('## Reviewer\n'):
            if review != '':
                if '## Paper Decision\n\n' in review:
                    review, paper_decision = review.split('## Paper Decision\n\n')[:2]
                    paper_decision = paper_decision.split('\n')[0]
                    paper_decision = 'Accept' if 'accept' in paper_decision.lower() else 'Reject'
                if '## Meta Review\n\n' in review:
                    review, meta_review = review.split('## Meta Review\n\n')[:2]
                reviews.append(review)

                def sec(tag, splitter='###'):
                    if f'### {tag}\n\n' in review:
                        return review.split(f'### {tag}\n\n')[1].split(splitter)[0]
                    return ''
                summary.append(sec('Summary'))
                soundness.append(sec('Soundness'))
                presentation.append(sec('Presentation'))
                contribution.append(sec('Contribution'))
                strengths.append(sec('Strengths'))
                weaknesses.append(sec('Weaknesses'))
                questions.append(sec('Questions'))
                flag_for_ethics_review.append(sec('Flag For Ethics Review'))
                r = sec('Rating')
                review_rate.append(r)
                rating.append(_rating_num(r) if r else 0)
                confidence.append(sec('Confidence', '******'))
        if paper_decision == '':
            return None
        pred['content'] = generated_text
        pred['reviews'] = reviews
        pred['summary'] = summary
        pred['review_rate'] = review_rate
        pred['rating'] = rating
        pred['soundness'] = soundness
        pred['presentation'] = presentation
        pred['contribution'] = contribution
        pred['strength'] = strengths
        pred['weaknesses'] = weaknesses
        pred['questions'] = questions
        pred['flag_for_ethics_review'] = flag_for_ethics_review
        pred['confidence'] = confidence
        pred['paper_decision'] = paper_decision
        pred['meta_review'] = meta_review
        pred['avg_rating'] = sum(rating) / len(rating)
        return pred
    except Exception:
        return None


def get_reviewer_score(generated_text):
    pred = get_reviewer_score_7B(generated_text)
    if pred is None:
        pred = get_reviewer_score_123B(generated_text)
    elif pred['rating'] == 0:      # original dead branch, kept for fidelity
        pred = get_reviewer_score_123B(generated_text)
    return pred


def valid_pred(pred):
    return (pred is not None and pred.get('paper_decision') in ('Accept', 'Reject')
            and pred.get('rating') and any(1 <= r <= 10 for r in pred['rating']))


# ---- compact review dict handed to the rewrite/loop ---------------------------
def to_review(pred):
    """Same top-level contract as B3/B3a reviews: 'Overall' (1-10) + 'Decision',
    plus the per-reviewer ICLR fields and the meta review. 'content' (the raw
    ~7k-token generation) stays in the log only."""
    n = len(pred['rating'])
    reviewers = []
    for i in range(n):
        def g(key):
            v = pred.get(key, [])
            return v[i].strip() if i < len(v) and isinstance(v[i], str) else ''
        reviewers.append({
            "Summary": g('summary'),
            "Soundness": g('soundness'),
            "Presentation": g('presentation'),
            "Contribution": g('contribution'),
            "Strengths": g('strength'),
            "Weaknesses": g('weaknesses'),
            "Questions": g('questions'),
            "Rating": g('review_rate'),
            "Confidence": g('confidence'),
        })
    return {
        "Reviews": reviewers,
        "Meta Review": pred.get('meta_review', '').strip(),
        "Overall": round(pred['avg_rating'], 2),
        "Decision": pred['paper_decision'],
    }


# ---- main entry: same interface as sakana_reviewer.perform_review -------------
def perform_review(paper_text, log=None):
    """Returns (review_dict | None, info) — info = {'level': .., 'n_valid': ..}
    (n_valid = number of parsed simulated reviewers, canonical 4)."""
    for level, pp_c in enumerate(TRUNC_LADDER):
        for attempt in range(RESAMPLE_PER_LEVEL):
            out, err = chat(paper_text[:pp_c])
            if err == "ctx":
                break                     # prompt too long -> next ladder level
            pred = get_reviewer_score(out)
            if not valid_pred(pred):
                continue                  # degenerate output -> resample
            review = to_review(pred)
            if log:
                log.write_text(json.dumps({
                    "level": level, "attempt": attempt,
                    "n_valid": len(pred['rating']),
                    "raw": pred['content'], "final": review}, indent=2))
            return review, {"level": level, "n_valid": len(pred['rating'])}
    return None, {"level": None, "n_valid": 0}


if __name__ == "__main__":
    # smoke: python3 cyclerev_reviewer.py <tex_file_or_dir>
    p = Path(sys.argv[1])
    if p.is_dir():
        text = "\n".join(f.read_text(errors="ignore")
                         for f in [p / "main.tex"] + sorted((p / "sections").glob("*.tex"))
                         if f.exists())
    else:
        text = p.read_text(errors="ignore")
    rev, info = perform_review(text)
    print(json.dumps(rev, indent=2)[:3000] if rev else "REVIEW FAILED")
    print(info)
