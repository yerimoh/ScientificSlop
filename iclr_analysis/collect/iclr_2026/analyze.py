"""
Analysis of the ICLR-2026 x Pangram corpus (n=19,490 submissions / 75,800 reviews).

Answers, at ~1300x the scale of the 15-paper famous-paper demo:
  Q1  How AI-written is the ICLR 2026 submission pool, and where does FARS sit on that scale?
  Q2  Does paper AI-ness predict rejection?  (accept rate by fraction_ai)
  Q3  Does paper AI-ness predict review scores, and which sub-score is it closest to?
  Q4  How AI-written are the reviews, and do AI reviews score differently than human ones?
  Q5  Do AI-written papers attract AI-written reviews?

Writes RESULTS.md + figs/.
"""
import json
import os
from collections import Counter

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# Pangram verdicts on the 15-paper demo (Evaluation/famous_paper_demo/pangram/pangram_scores.json)
FARS_DEMO = (63.9, 85.1)     # fully-agent-written papers, % of doc AI
LANDMARK_DEMO = (0.0, 7.0)   # pre-LLM landmark papers (Word2Vec/Attention/BERT/...)

PRED_ORDER = ["Fully human-written", "Lightly AI-edited", "Moderately AI-edited",
              "Heavily AI-edited", "Fully AI-generated"]


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name))]


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _z(xs):
    v = np.asarray(xs, dtype=float)
    return (v - v.mean()) / (v.std() or 1)


def _logit_fit(X, y, iters=60):
    """Newton-Raphson logistic regression; small and dependency-free."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def q(xs, name, out):
    xs = np.asarray(xs, dtype=float)
    out.append(f"| {name} | {len(xs)} | {xs.mean():.4f} | {np.median(xs):.4f} | "
               f"{np.percentile(xs,75):.4f} | {np.percentile(xs,90):.4f} | {np.percentile(xs,99):.4f} | {xs.max():.4f} |")


def main():
    papers = load("papers_2026.jsonl")
    reviews = load("reviews_2026.jsonl")
    P = [p for p in papers if p["fraction_ai"] is not None]
    out = []
    A = out.append

    A("# ICLR 2026 x Pangram — scaled AI-authorship measurement\n")
    A(f"Corpus: **{len(papers):,} submissions**, {len(reviews):,} reviews. "
      f"Pangram verdicts on **{len(P):,}** submissions ({100*len(P)/len(papers):.1f}%). "
      f"Accepted {sum(p['accept'] for p in papers):,} "
      f"({100*sum(p['accept'] for p in papers)/len(papers):.1f}%).\n")
    A("Source: Pangram was the official AI-detection partner of ICLR 2026 and published per-submission "
      "and per-review verdicts at `iclr.pangram.com`. `fraction_ai` = fraction of the **full submission "
      "text** Pangram classifies as AI-generated. No API credits were needed.\n")

    # ---------- Q1: distribution ----------
    fa = np.array([p["fraction_ai"] for p in P])
    A("## Q1 — How AI-written is the ICLR 2026 pool?\n")
    A("| set | n | mean | median | p75 | p90 | p99 | max |")
    A("|---|---|---|---|---|---|---|---|")
    q(fa, "all submissions", out)
    q([p["fraction_ai"] for p in P if p["accept"]], "accepted", out)
    q([p["fraction_ai"] for p in P if not p["accept"]], "rejected", out)
    A("")
    bands = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.25), (0.25, 0.50), (0.50, 0.639), (0.639, 1.01)]
    A("| fraction_ai band | n | share |")
    A("|---|---|---|")
    for lo, hi in bands:
        n = int(((fa > lo) & (fa <= hi)).sum()) if lo != hi else int((fa == 0).sum())
        lab = "exactly 0 (fully human)" if lo == hi else (
            f"> {lo:g} .. {hi:g}" + ("  ← FARS range (fully-agent papers)" if lo == 0.639 else ""))
        A(f"| {lab} | {n:,} | {100*n/len(fa):.1f}% |")
    A("")
    n_fars = int((fa >= FARS_DEMO[0] / 100).sum())
    A(f"**Anchors from the 15-paper demo:** fully-agent FARS papers scored {FARS_DEMO[0]}–{FARS_DEMO[1]}% AI; "
      f"pre-LLM landmarks {LANDMARK_DEMO[0]}–{LANDMARK_DEMO[1]}%. "
      f"In ICLR 2026, **{n_fars:,} submissions ({100*n_fars/len(fa):.2f}%) reach the FARS band** "
      f"(≥{FARS_DEMO[0]}% AI), while {int((fa<=0.07).sum()):,} ({100*(fa<=0.07).mean():.1f}%) sit in the landmark band (≤7%).\n")

    # ---------- Q2: accept/reject ----------
    A("## Q2 — Does paper AI-ness predict rejection?\n")
    acc = np.array([p["accept"] for p in P], dtype=bool)
    u, pu = stats.mannwhitneyu(fa[acc], fa[~acc], alternative="two-sided")
    d = (fa[~acc].mean() - fa[acc].mean()) / np.sqrt((fa[acc].var() + fa[~acc].var()) / 2)
    A(f"accepted mean {fa[acc].mean():.4f} (n={acc.sum():,}) vs rejected mean {fa[~acc].mean():.4f} "
      f"(n={(~acc).sum():,}); Mann-Whitney p={fmt_p(pu)}, Cohen's d={d:+.3f} (rejected − accepted).\n")
    A("| fraction_ai band | n | accept rate |")
    A("|---|---|---|")
    edges = [0.0, 1e-9, 0.05, 0.15, 0.30, 0.50, 1.01]
    labels = ["= 0", "0–0.05", "0.05–0.15", "0.15–0.30", "0.30–0.50", "> 0.50"]
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        m = (fa >= lo) & (fa < hi) if lo == 0.0 else (fa >= lo) & (fa < hi)
        if lab == "= 0":
            m = fa == 0
        if m.sum() < 20:
            continue
        A(f"| {lab} | {int(m.sum()):,} | {100*acc[m].mean():.1f}% |")
    A("")

    # ---------- Q3: scores ----------
    A("## Q3 — Paper AI-ness vs review scores\n")
    A("| score | n | Spearman rho vs fraction_ai | p |")
    A("|---|---|---|---|")
    for key in ["rating_mean", "soundness_mean", "presentation_mean", "contribution_mean", "confidence_mean"]:
        pair = [(p["fraction_ai"], p[key]) for p in P if p.get(key) is not None]
        x, y = np.array([a for a, _ in pair]), np.array([b for _, b in pair])
        r, pv = stats.spearmanr(x, y)
        A(f"| {key} | {len(x):,} | {r:+.4f} | {fmt_p(pv)} |")
    A("")
    A("| fraction_ai band | n | mean rating | mean soundness | mean presentation | mean contribution |")
    A("|---|---|---|---|---|---|")
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        m = [p for p in P if (p["fraction_ai"] == 0 if lab == "= 0" else lo <= p["fraction_ai"] < hi)]
        if len(m) < 20:
            continue
        def mm(k):
            v = [p[k] for p in m if p.get(k) is not None]
            return f"{np.mean(v):.3f}" if v else "-"
        A(f"| {lab} | {len(m):,} | {mm('rating_mean')} | {mm('soundness_mean')} | {mm('presentation_mean')} | {mm('contribution_mean')} |")
    A("")

    # ---------- Q4: reviews ----------
    A("## Q4 — How AI-written are the reviews?\n")
    c = Counter(r["prediction"] for r in reviews)
    A("| Pangram verdict on review | n | share |")
    A("|---|---|---|")
    for k in PRED_ORDER:
        A(f"| {k} | {c[k]:,} | {100*c[k]/len(reviews):.1f}% |")
    A(f"\n**{100*(len(reviews)-c['Fully human-written'])/len(reviews):.1f}% of ICLR 2026 reviews carry AI text**; "
      f"{100*c['Fully AI-generated']/len(reviews):.1f}% are fully AI-generated.\n")

    A("| review class | n | mean rating | mean confidence | mean soundness | mean chars |")
    A("|---|---|---|---|---|---|")
    for k in PRED_ORDER:
        g = [r for r in reviews if r["prediction"] == k]
        def mm(key):
            v = [r[key] for r in g if r.get(key) is not None]
            return f"{np.mean(v):.3f}" if v else "-"
        A(f"| {k} | {len(g):,} | {mm('rating')} | {mm('confidence')} | {mm('soundness')} | {mm('n_chars')} |")
    A("")
    hu = np.array([r["rating"] for r in reviews if r["prediction"] == "Fully human-written" and r["rating"] is not None])
    ai = np.array([r["rating"] for r in reviews if r["prediction"] == "Fully AI-generated" and r["rating"] is not None])
    u, pv = stats.mannwhitneyu(ai, hu, alternative="two-sided")
    A(f"Fully-AI reviews give **{ai.mean()-hu.mean():+.3f}** rating vs fully-human reviews "
      f"({ai.mean():.3f} vs {hu.mean():.3f}, Mann-Whitney p={fmt_p(pv)}).\n")

    # ---------- Q5: paper AI-ness x review AI-ness ----------
    A("## Q5 — Do AI-written papers attract AI-written reviews?\n")
    pair = [(p["fraction_ai"], p["review_ai_frac"]) for p in P if p.get("review_ai_frac") is not None]
    x, y = np.array([a for a, _ in pair]), np.array([b for _, b in pair])
    r, pv = stats.spearmanr(x, y)
    A(f"Spearman(paper fraction_ai, share of its reviews containing AI text) = **{r:+.4f}** "
      f"(n={len(x):,}, p={fmt_p(pv)}).\n")
    A("| paper fraction_ai band | n | mean share of AI-touched reviews | mean share of fully-AI reviews |")
    A("|---|---|---|---|")
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        m = [p for p in P if (p["fraction_ai"] == 0 if lab == "= 0" else lo <= p["fraction_ai"] < hi)
             and p.get("review_ai_frac") is not None]
        if len(m) < 20:
            continue
        A(f"| {lab} | {len(m):,} | {np.mean([p['review_ai_frac'] for p in m]):.3f} | "
          f"{np.mean([p['review_fully_ai_frac'] for p in m]):.3f} |")
    A("")

    # ---------- Q6: is it just that AI papers are worse papers? ----------
    A("## Q6 — Does AI-ness cost anything *beyond* the review scores?\n")
    A("Decisions are made from the reviews, so most of the effect in Q2 should be mediated by "
      "rating. Holding rating fixed isolates whatever is left.\n")
    A("| rating band | n | accept rate, fraction_ai = 0 | accept rate, fraction_ai > 0.30 | gap |")
    A("|---|---|---|---|---|")
    for lo, hi in [(0, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 11)]:
        g = [p for p in P if p["rating_mean"] is not None and lo <= p["rating_mean"] < hi]
        a0 = [p["accept"] for p in g if p["fraction_ai"] == 0]
        a1 = [p["accept"] for p in g if p["fraction_ai"] > 0.30]
        if len(a0) < 30 or len(a1) < 30:
            continue
        A(f"| {lo}–{hi} | {len(g):,} | {100*np.mean(a0):.1f}% (n={len(a0):,}) | "
          f"{100*np.mean(a1):.1f}% (n={len(a1):,}) | {100*(np.mean(a1)-np.mean(a0)):+.1f} pp |")
    A("")
    # logistic regression accept ~ rating + fraction_ai (standardized)
    g = [p for p in P if p["rating_mean"] is not None]
    X = np.column_stack([np.ones(len(g)),
                         _z([p["rating_mean"] for p in g]),
                         _z([p["fraction_ai"] for p in g])])
    y = np.array([p["accept"] for p in g], dtype=float)
    beta = _logit_fit(X, y)
    A(f"Logistic regression on {len(g):,} submissions, both predictors standardized: "
      f"**accept ~ {beta[1]:+.3f}·rating {beta[2]:+.3f}·fraction_ai** (log-odds per SD). "
      f"Rating carries {abs(beta[1]/beta[2]):.1f}× the weight of AI fraction.\n")

    # ---------- Q7: reviewer disagreement ----------
    A("## Q7 — Do AI-written papers split their reviewers?\n")
    g = [p for p in P if p["n_reviews"] and p["n_reviews"] >= 3
         and p["rating_max"] is not None and p["rating_min"] is not None]
    spread = np.array([p["rating_max"] - p["rating_min"] for p in g])
    fag = np.array([p["fraction_ai"] for p in g])
    r, pv = stats.spearmanr(fag, spread)
    A(f"Spearman(fraction_ai, max−min rating within a paper) = **{r:+.4f}** "
      f"(n={len(g):,}, p={fmt_p(pv)}); mean spread {spread.mean():.3f}.\n")

    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
