#!/usr/bin/env python3
"""evidence_gap  -  The gap between the evidence a paper aggregates and the evidence it
displays (Artifacts plane).

An empirical paper (one with a body result table) aggregates results over discrete instances;
the item finds every instance the paper actually DISPLAYS and records the gap when there is none.

Exhibits, from the tex alone (appendix included; existence, not placement):
    env       a display environment carrying consumed/produced material (verbatim, lstlisting,
              minted, quoting, tcolorbox, mdframed, exampleblock, promptbox)
    caption   a figure/table caption announcing an example, a case or a failure
    quote     a long quoted passage (>=40 chars between quotation marks) outside Related Work
Sub-observation: a section titled error/failure/case-study/qualitative analysis whose span holds
zero exhibits (recorded with word count and percentage-token count; located, not scored).

Verdicts: exhibited (one per exhibit) / titled_without_exhibit (located observation) /
unexhibited (once per applicable paper with no exhibit anywhere). Observational item; no paper
label (the xsec_ref rule: absence is an observation, not a violation).

Usage: python3 measure.py [--corpus AI|HU|all] [--limit N] [--only IDS]
       -> ../results/{instances.jsonl,papers.jsonl,summary.json}
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import load_doc, prose_view
from tables import parse_tables
from records import instance, write_jsonl, VIOLATION, CONSISTENT

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "evidence_gap_v1"

# 0920. Add the standard quote environment. The list only had the package-provided quoting and was missing the most common one, quote.
# Re-measured on the 165 papers, AI is unchanged and only the human side rises from 28 to 34 papers. Opening the newly flagged human papers,
# they are indeed the exhibits this item looks for: Chain-of-Thought failure-case Q&A, full review prompts, and the like.
ENV = re.compile(r"\\begin\{(verbatim|lstlisting|minted|quoting|quote|tcolorbox|mdframed|exampleblock|promptbox|examplebox)\}", re.I)
CAPTION = re.compile(r"\\caption\{[^{}]{0,160}\b(an example|example of|examples? (?:of|from|where)|case stud\w*|failure case|qualitative (?:example|result|analysis)|sample (?:output|response|generation))\b", re.I)
QUOTE = re.compile(r"``[^`']{40,600}''", re.S)
TITLED = re.compile(r"\\((?:sub)+section|section)\*?\{([^{}]*(?:error analysis|failure (?:analysis|mode|case)|case stud\w*|qualitative analysis|worked example)[^{}]*)\}", re.I)
RELATED = re.compile(r"\\section\*?\{[^}]*related[^}]*\}", re.I)
PCT = re.compile(r"\d+(?:\.\d+)?\s*\\?%")


def is_result_table(t) -> bool:
    ncell = sum(1 for r in t["rows"] for c in r["cells"] if c.get("kind") == "number")
    return len(t["rows"]) >= 2 and ncell >= 4


def find_exhibits(tex: str, doc) -> list[tuple[str, int, str]]:
    """Every displayed instance: (kind, position, excerpt)."""
    rw = RELATED.search(tex, doc.doc_start, doc.doc_end)
    rw_span = None
    if rw:
        nxt = tex.find("\\section", rw.end())
        rw_span = (rw.start(), nxt if nxt > 0 else doc.doc_end)
    out = []
    for m in ENV.finditer(tex):
        out.append(("env", m.start(), m.group(1)))
    for m in CAPTION.finditer(tex):
        out.append(("caption", m.start(), m.group(1)))
    for m in QUOTE.finditer(tex):
        if rw_span and rw_span[0] <= m.start() < rw_span[1]:
            continue                      # quoted material in Related Work belongs to other papers
        out.append(("quote", m.start(), m.group(0)[:80]))
    return sorted(out, key=lambda x: x[1])


def measure(paper):
    doc = load_doc(paper)
    tex = doc.tex
    secs = {s.idx: s for s in doc.top_sections}
    body_tabs = [t for t in parse_tables(tex, doc.doc_start, doc.doc_end)
                 if is_result_table(t)
                 and not (secs.get(doc.top_of(t["start"])) and secs[doc.top_of(t["start"])].in_appendix)]
    if not body_tabs:
        row = {"corpus": paper["corpus"], "id": paper["id"], "applicable": False,
               "n_exhibits": 0, "slop_score": None, "slop_numerator": None,
               "slop_denominator": None, "coverage": None, "label": "NA"}
        return row, []
    words = len(prose_view(tex[doc.doc_start:doc.doc_end], base=doc.doc_start).text.split())
    app0 = min([s.start for s in doc.top_sections if s.in_appendix], default=doc.doc_end)
    exhibits = find_exhibits(tex, doc)
    insts, k = [], 0
    for kind, pos, what in exhibits:
        k += 1
        insts.append(instance(paper["id"], "evidence_gap", k, CONSISTENT,
                              {"kind": "exhibit", "exhibit_kind": kind, "span": [pos, pos + 40],
                               "in_appendix": pos >= app0},
                              {"kind": "text", "excerpt": str(what)[:120]},
                              checker=CHECKER, verdict="exhibited"))
    titled_empty = []
    for m in TITLED.finditer(tex):
        end = tex.find("\\section", m.end())
        seg_end = end if end > 0 else min(m.end() + 8000, doc.doc_end)
        if any(m.start() <= pos < seg_end for _, pos, _ in exhibits):
            continue
        seg = tex[m.end():seg_end]
        titled_empty.append({"title": m.group(2)[:60], "words": len(seg.split()),
                             "percent_tokens": len(PCT.findall(seg))})
        k += 1
        insts.append(instance(paper["id"], "evidence_gap", k, VIOLATION,
                              {"kind": "section", "title": m.group(2)[:80], "span": [m.start(), seg_end]},
                              {"kind": "text", "words": len(seg.split()),
                               "percent_tokens": len(PCT.findall(seg))},
                              checker=CHECKER, verdict="titled_without_exhibit",
                              note="a section named for cases holds percentages and not one case (observational)"))
    if not exhibits:
        k += 1
        insts.append(instance(paper["id"], "evidence_gap", k, VIOLATION,
                              {"kind": "paper", "span": [doc.doc_start, doc.doc_end]},
                              {"kind": "text", "n_body_result_tables": len(body_tabs)},
                              checker=CHECKER, verdict="unexhibited",
                              note="aggregate results displayed; not one instance exhibited (observational)"))
    row = {"corpus": paper["corpus"], "id": paper["id"], "applicable": True,
           "n_exhibits": len(exhibits),
           "exhibit_kinds": dict(Counter(kind for kind, _, _ in exhibits)),
           "exhibits_appendix_only": bool(exhibits) and all(pos >= app0 for _, pos, _ in exhibits),
           "exhibits_per10k": round(len(exhibits) / max(1, words) * 10000, 2),
           "titled_without_exhibit": titled_empty[:6],
           "slop_score": 0.0 if exhibits else 1.0,
           "slop_numerator": 0 if exhibits else 1, "slop_denominator": 1,
           "coverage": 1.0, "label": "observational"}
    return row, insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all", choices=["AI", "HU", "all"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--only", default="")
    a = ap.parse_args()
    papers = ([] if a.corpus == "HU" else ai_papers()) + ([] if a.corpus == "AI" else hu_papers())
    if a.only:
        ids = set(a.only.split(",")); papers = [p for p in papers if p["id"] in ids]
    if a.limit:
        papers = papers[:a.limit]
    rows, insts, fails = [], [], []
    for p in papers:
        try:
            r, ii = measure(p); rows.append(r); insts.extend(ii)
        except Exception as ex:
            fails.append({"id": p["id"], "error": repr(ex)})
    write_jsonl(os.path.join(RESULTS, "papers.jsonl"), rows)
    write_jsonl(os.path.join(RESULTS, "instances.jsonl"), insts)
    def agg(cor):
        rs = [r for r in rows if r["corpus"] == cor and r["applicable"]]
        return {"papers": len(rs),
                "unexhibited": sum(1 for r in rs if r["slop_score"] == 1.0),
                "exhibited": sum(1 for r in rs if r["slop_score"] == 0.0),
                "titled_without_exhibit": sum(len(r["titled_without_exhibit"]) for r in rs),
                "appendix_only": sum(1 for r in rs if r.get("exhibits_appendix_only"))}
    summary = {"checker": CHECKER, "n_papers": len(rows), "failures": fails,
               "by_corpus": {c: agg(c) for c in ("AI", "HU")},
               "note": "observational; no paper label; the graded observation is exhibits_per10k, "
                       "never mixed into the score; report the human base rate next to any AI rate"}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
