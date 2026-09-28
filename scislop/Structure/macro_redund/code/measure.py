#!/usr/bin/env python3
"""macro_redund  -  Cross-section sentence recycling (Structure plane).

The unit is a SENTENCE of the paper. A sentence is recycled when at least half of its tokens
are covered by n-grams (n = 8) that already occurred in an earlier, different top-level section.
The score is the share of recycled sentences among the sentences long enough to be checked
(slop_definition_revised_0909.md 1.2, as revised on 0911: repetition is measured, not judged).

Judging whether a repeat was "necessary" was rejected (versions 2 and 3 in _archive/). Version 2
exempted whole sections by their name; version 3 replaced that with a hand-written stack of cue
lists and five thresholds, which an NLI probe showed to be wrong on about a quarter of its
verdicts. The measurable fact is how much of a paper is text that an earlier section already
contained. That fact is reported in two strata, body sections and summary slots (abstract,
conclusion), so the genre argument "a conclusion is supposed to restate" is answered with a
number rather than an exemption.

Definitions
  eligible sentence   at least MIN_TOK tokens (shorter sentences have no n-gram of length N)
  covered token       a token inside an N-gram of the sentence that also occurs in an earlier unit
                      of a different section (sentinel-bearing n-grams never count)
  r(s)                covered tokens / tokens of the sentence
  recycled sentence   eligible and r(s) >= TAU
  slop_score          recycled / eligible          coverage = eligible / all sentences
  slop_score_agg      min(1, slop_score / CEIL)   the registered ceiling used only where items
                      are averaged (SLOP_SCORE.md); 1 = one sentence in ten is recycled

Usage:  python3 measure.py [--limit N] [--only ids] [--out DIR] [--n 8] [--tau 0.5] [--cliche-min K]
        -> <out>/{papers.jsonl, sentences.jsonl, summary.json}     (default out = ../results)
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import load_doc, prose_view, sentences, tokens, clean_title, SENTINELS
from records import write_jsonl, compare_groups, rate, describe, slop_score

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "macro_redund_v4"

# ----------------------------------------------------------------------------- registered parameters
P = dict(
    N=8,            # n-gram length; 4 admits stock phrases, 12 misses one-word rewrites (Winnowing noise threshold)
    TAU=0.5,        # a sentence is recycled when at least this share of its tokens is covered
    MIN_TOK=8,      # sentences shorter than N cannot be checked; they lower coverage
    CEIL=0.10,      # ceiling for the aggregation score: one recycled sentence in ten = 1
    CLICHE_MIN=0,   # optional: drop n-grams found in >= K human papers (0 = off; at n = 8 no gram reaches 10 papers)
)

SUMMARY_SLOT_RE = re.compile(r"conclu|concluding|\bsummary\b|closing remarks|final remarks", re.I)
STATEMENT_SLOT_RE = re.compile(r"ethic|impact statement|broader impact|societal|reproducibility statement|checklist|acknowledg", re.I)
HEADING_RE = re.compile(r"\\(?:paragraph|subparagraph)\*?\s*(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}")


def slot_of(title: str, role: str) -> str:
    if role == "Abstract" or SUMMARY_SLOT_RE.search(title or ""):
        return "summary"
    if STATEMENT_SLOT_RE.search(title or ""):
        return "statement"
    return "body"


def grams(low, n):
    return [tuple(low[i:i + n]) for i in range(len(low) - n + 1)]


# ----------------------------------------------------------------------------- units

def collect_units(doc):
    """Sentences of the abstract (pseudo-section -1) and of every body section, in document order.
    Headings (which the prose view prints as one-line sentences) and statement sections are dropped."""
    headings = {clean_title(x.title).lower().rstrip(". ") for x in doc.sections}
    headings |= {clean_title(m.group(1)).lower().rstrip(". ") for m in HEADING_RE.finditer(doc.tex)}
    headings.discard("")
    units, sec_info = [], {}
    if doc.abstract.strip():
        am = re.search(r"\\begin\{abstract\}", doc.tex)
        mt = prose_view(doc.abstract, base=am.end() if am else 0)
        sec_info[-1] = {"title": "Abstract", "role": "Abstract", "slot": "summary"}
        for s in sentences(mt):
            s.update({"sec": -1, "role": "Abstract", "title": "Abstract", "slot": "summary"}); units.append(s)
    for sec in doc.body_sections():
        slot = slot_of(sec.title, sec.role)
        sec_info[sec.idx] = {"title": sec.title, "role": sec.role, "slot": slot}
        if slot == "statement":
            continue
        mt = prose_view(doc.tex[sec.body_start:sec.end], base=sec.body_start)
        for s in sentences(mt):
            if s["text"].lower().rstrip(". ") in headings:
                continue
            s.update({"sec": sec.idx, "role": sec.role, "title": sec.title, "slot": slot}); units.append(s)
    units.sort(key=lambda u: u["tex_span"][0])
    for u in units:
        u["toks"] = tokens(u["text"]); u["lower"] = [t.lower() for t in u["toks"]]
    return units, sec_info


# ----------------------------------------------------------------------------- score

def build_cliche_set(papers, k, n):
    """n-grams present in at least k distinct papers of the given list (genre stock phrases)."""
    cnt = Counter()
    for p in papers:
        try:
            units, _ = collect_units(load_doc(p))
        except Exception:
            continue
        seen = set()
        for u in units:
            for g in grams(u["lower"], n):
                if not (SENTINELS & set(g)):
                    seen.add(g)
        cnt.update(seen)
    return {g for g, c in cnt.items() if c >= k}


def measure(paper, cliche=frozenset()):
    doc = load_doc(paper)
    units, sec_info = collect_units(doc)
    n, tau = P["N"], P["TAU"]
    # first occurrence of every n-gram, in document order: that unit is the original
    first = {}
    for ui, u in enumerate(units):
        for g in grams(u["lower"], n):
            if SENTINELS & set(g) or g in cliche:
                continue
            first.setdefault(g, ui)
    rows = []
    for ui, u in enumerate(units):
        n_tok = len(u["toks"]); covered = set(); sources = Counter()
        for i, g in enumerate(grams(u["lower"], n)):
            if SENTINELS & set(g) or g in cliche:
                continue
            f = first.get(g)
            if f is not None and f < ui and units[f]["sec"] != u["sec"]:
                covered.update(range(i, i + n)); sources[f] += 1
        r = len(covered) / n_tok if n_tok else 0.0
        run = best = 0
        for i in range(n_tok):
            run = run + 1 if i in covered else 0; best = max(best, run)
        eligible = n_tok >= P["MIN_TOK"]
        src = sources.most_common(1)[0][0] if sources else None
        rows.append({"corpus": paper["corpus"], "id": paper["id"], "unit": ui, "section": u["title"], "role": u["role"], "slot": u["slot"],
                     "span": u["tex_span"], "n_tokens": n_tok, "covered": len(covered), "r": round(r, 3), "longest_run": best,
                     "eligible": eligible, "recycled": bool(eligible and r >= tau),
                     "source_section": units[src]["title"] if src is not None else None, "source_role": units[src]["role"] if src is not None else None,
                     "source_slot": units[src]["slot"] if src is not None else None,
                     "text": u["text"][:400], "source_text": units[src]["text"][:400] if src is not None else None})
    elig = [x for x in rows if x["eligible"]]
    rec = [x for x in elig if x["recycled"]]
    n_words = sum(x["n_tokens"] for x in rows) or 1

    def share(xs):
        e = [x for x in xs if x["eligible"]]
        return rate(sum(1 for x in e if x["recycled"]), len(e))

    body = [x for x in rows if x["slot"] == "body"]; summ = [x for x in rows if x["slot"] == "summary"]
    sc = slop_score(len(rec), len(elig))
    row = {"corpus": paper["corpus"], "id": paper["id"], "source_incomplete": bool(doc.report["missing"]),
           "n_sentences": len(rows), "n_eligible": len(elig), "n_recycled": len(rec), "n_words": n_words,
           "n_sections": len(sec_info), "n_body_sections": sum(1 for i in sec_info.values() if i["slot"] == "body"),
           **sc, "coverage": rate(len(elig), len(rows)),
           "slop_score_agg": None if sc["slop_score"] is None else round(min(1.0, sc["slop_score"] / P["CEIL"]), 4),
           "slop_body": share(body), "slop_summary": share(summ),
           "n_eligible_body": sum(1 for x in body if x["eligible"]), "n_eligible_summary": sum(1 for x in summ if x["eligible"]),
           "repeat_word_rate": round(sum(x["covered"] for x in rows) / n_words, 4),
           "mean_r": round(sum(x["r"] for x in elig) / len(elig), 4) if elig else None,
           "longest_repeat": max((x["longest_run"] for x in rows), default=0),
           "has_recycled": bool(rec),
           "recycled_by_role": dict(Counter(x["role"] for x in rec)),
           "recycled_by_pair": dict(Counter(f'{x["source_role"]}>{x["role"]}' for x in rec))}
    return row, rows


# ----------------------------------------------------------------------------- corpus run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of ids")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--n", type=int, default=None, help="override N (sensitivity: 6, 8, 10)")
    ap.add_argument("--tau", type=float, default=None, help="override TAU (sensitivity: 0.4, 0.5, 0.6)")
    ap.add_argument("--cliche-min", type=int, default=None, help="drop n-grams found in >= K human papers")
    a = ap.parse_args()
    if a.n is not None: P["N"] = a.n
    if a.tau is not None: P["TAU"] = a.tau
    if a.cliche_min is not None: P["CLICHE_MIN"] = a.cliche_min
    papers = ai_papers() + hu_papers()
    cliche = build_cliche_set(hu_papers(), P["CLICHE_MIN"], P["N"]) if P["CLICHE_MIN"] else frozenset()
    if a.only:
        ids = set(a.only.split(",")); papers = [p for p in papers if p["id"] in ids]
    if a.limit:
        papers = papers[:a.limit]
    rows, sents, fails = [], [], []
    for p in papers:
        try:
            r, ss = measure(p, cliche); rows.append(r); sents.extend(ss)
        except Exception as ex:
            fails.append({"id": p["id"], "error": repr(ex)[:300]})
    os.makedirs(a.out, exist_ok=True)
    write_jsonl(os.path.join(a.out, "papers.jsonl"), rows)
    write_jsonl(os.path.join(a.out, "sentences.jsonl"), [s for s in sents if s["covered"] > 0])
    ai = [r for r in rows if r["corpus"] == "AI"]
    hu = [r for r in rows if r["corpus"] == "HU" and not r["source_incomplete"]]
    hu_all = [r for r in rows if r["corpus"] == "HU"]
    summary = {"checker": CHECKER, "params": dict(P), "n_cliche_grams": len(cliche), "n_ai": len(ai), "n_hu_complete": len(hu),
               "n_hu_incomplete_source": len(hu_all) - len(hu), "failures": fails,
               "unit": "sentence with >= MIN_TOK tokens", "score": "recycled sentences / eligible sentences; recycled = r(s) >= TAU",
               "metrics": {}}
    for m in ["slop_score", "slop_score_agg", "slop_body", "slop_summary", "coverage", "repeat_word_rate", "mean_r", "longest_repeat",
              "n_recycled", "n_eligible", "n_sentences", "n_words", "n_sections"]:
        summary["metrics"][m] = compare_groups(m, [r[m] for r in ai if r.get(m) is not None], [r[m] for r in hu if r.get(m) is not None])
    summary["has_recycled_share"] = {"AI": rate(sum(1 for r in ai if r["has_recycled"]), len(ai)), "HU": rate(sum(1 for r in hu if r["has_recycled"]), len(hu))}
    summary["quantiles_slop_score"] = {c: {str(q): describe([r["slop_score"] for r in g])} for c, g in (("AI", ai), ("HU", hu)) for q in ("all",)}
    import numpy as np
    summary["quantiles_slop_score"] = {c: {f"q{q}": round(float(np.percentile([r["slop_score"] for r in g if r["slop_score"] is not None], q)), 4) for q in (0, 10, 25, 50, 75, 90, 100)}
                                       for c, g in (("AI", ai), ("HU", hu)) if g}
    summary["recycled_by_role"] = {c: dict(sum((Counter(r["recycled_by_role"]) for r in g), Counter())) for c, g in (("AI", ai), ("HU", hu))}
    summary["recycled_by_pair"] = {c: dict(sum((Counter(r["recycled_by_pair"]) for r in g), Counter())) for c, g in (("AI", ai), ("HU", hu))}
    strata = {}
    for lo, hi in [(0, 120), (120, 200), (200, 300), (300, 10 ** 6)]:
        strata[f"sentences_{lo}_{hi}"] = {c: describe([r["slop_score"] for r in g if lo <= r["n_eligible"] < hi]) for c, g in (("AI", ai), ("HU", hu))}
    summary["strata_by_eligible_sentences"] = strata
    from scipy.stats import spearmanr
    summary["confound_spearman"] = {}
    for c, g in (("AI", ai), ("HU", hu)):
        for f in ["n_words", "n_eligible", "n_sections"]:
            pts = [(r[f], r["slop_score"]) for r in g if r["slop_score"] is not None]
            rho = spearmanr([x for x, _ in pts], [y for _, y in pts]).correlation if len(pts) > 3 else None
            summary["confound_spearman"][f"{c}:slop_score~{f}"] = None if rho is None or rho != rho else round(float(rho), 3)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ["n_ai", "n_hu_complete", "n_cliche_grams", "has_recycled_share", "quantiles_slop_score"]}, indent=1))
    for m in ["slop_score", "slop_score_agg", "slop_body", "slop_summary", "coverage", "repeat_word_rate", "mean_r", "longest_repeat"]:
        s = summary["metrics"][m]
        print(f"  {m:18} AI mean {s['AI'].get('mean')} med {s['AI'].get('median')} | HU mean {s['HU'].get('mean')} med {s['HU'].get('median')} | cliff {s['cliff_delta_AI_minus_HU']} p {s['mwu_p']}")
    print("  confound spearman:", summary["confound_spearman"])
    if fails:
        print("FAILURES", fails)


if __name__ == "__main__":
    main()
