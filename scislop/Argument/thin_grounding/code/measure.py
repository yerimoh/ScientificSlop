#!/usr/bin/env python3
"""thin_grounding  -  How much prior work the paper engages (Argument plane), v1.

Counts how many distinct works the body cites. AI papers tend to cite in the expected places but draw
on few works: on SciSlopBench the AI paper cites a median of 16 distinct works, its human anchor 48.

Unit = top-level body section (appendix and acknowledgements excluded, >= MIN_SECTION_WORDS prose words).
A section fails when it cites no work.
    slop_score = sections citing no work / body sections                     (weak below 3 sections)
Graded observation, reported next to the score and not mixed into it:
    works_per_1k = distinct cited works in the body / body prose words * 1000   (lower = thinner)

Deterministic: reads \\cite keys and the prose view only. A key cited twice counts once; keys inside
captions count. `extraction_suspect` marks papers with no \\cite at all or with 3+ raw citation markers
left outside \\cite (a PDF rebuild that did not convert them), so a lost citation is not read as a
missing one.

Usage: python3 measure.py [--corpus AI|HU|all] [--limit N] [--only IDS]
       -> ../results/{instances.jsonl,papers.jsonl,summary.json}
"""
from __future__ import annotations
import argparse, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import load_doc, prose_view, CITE_RE
from records import instance, write_jsonl, slop_score, compare_groups, VIOLATION, CONSISTENT

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "thin_grounding_v1"
MIN_SECTION_WORDS = 80
# raw citation markers left outside \cite: a PDF rebuild that missed them makes a cited paper look uncited
RAW_MARKER = re.compile(r"\[\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*\]|\([A-Z][A-Za-z\-]+(?: et al\.?| and [A-Z][a-z]+| & [A-Z][a-z]+)?,? (?:19|20)\d\d[a-z]?(?:;[^)]*)?\)"
                        r"|[A-Z][a-z]+ et al\.,? \(?(?:19|20)\d\d")


def keys_in(text: str) -> list[str]:
    return [k.strip() for m in CITE_RE.finditer(text) for k in re.split(r"[,\s]+", m.group(1)) if k.strip()]


def measure(paper):
    doc = load_doc(paper)
    tex = doc.tex
    body = doc.body_sections()
    body_end = doc.body_end()
    words = len(prose_view(tex[doc.doc_start:body_end], base=doc.doc_start).text.split())
    insts, k, checked, failed, seen = [], 0, 0, 0, []
    per_section = []
    for s in body:
        seg = tex[s.body_start:s.end]
        sw = len(prose_view(seg, base=s.body_start).text.split())
        if sw < MIN_SECTION_WORDS:
            continue
        ks = keys_in(seg)
        new = [x for x in dict.fromkeys(ks) if x not in seen]
        seen += new
        checked += 1
        status = CONSISTENT if ks else VIOLATION
        failed += status == VIOLATION
        k += 1
        insts.append(instance(paper["id"], "thin_grounding", k, status,
                              {"kind": "section", "title": s.title[:80], "role": s.role, "span": [s.start, s.end]},
                              {"kind": "citations", "n_citations": len(ks), "n_works": len(set(ks)), "n_new_works": len(new)},
                              checker=CHECKER, verdict="uncited_section" if status == VIOLATION else "cites",
                              section_words=sw))
        per_section.append({"title": s.title[:60], "role": s.role, "words": sw, "works": len(set(ks)), "new_works": len(new)})
    all_keys = keys_in(tex[doc.doc_start:body_end])
    works = len(set(all_keys))
    raw = len(RAW_MARKER.findall(CITE_RE.sub(" ", tex[doc.doc_start:body_end])))
    row = {"corpus": paper["corpus"], "id": paper["id"],
           **slop_score(failed, checked),
           "coverage": 1.0 if checked else None,
           "n_body_sections": checked, "n_uncited_sections": failed,
           "n_works": works, "n_citations": len(all_keys), "body_words": words,
           "works_per_1k": round(works / max(1, words) * 1000, 3),
           "citations_per_work": round(len(all_keys) / works, 3) if works else None,
           "no_cite_commands": works == 0, "raw_markers_outside_cite": raw,
           "extraction_suspect": works == 0 or raw >= 3,
           "sections": per_section, "label": "observational"}
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
    by = lambda cor, key: [r[key] for r in rows if r["corpus"] == cor and r.get(key) is not None]
    summary = {"checker": CHECKER, "n_papers": len(rows), "failures": fails,
               "slop_score": compare_groups("slop_score", by("AI", "slop_score"), by("HU", "slop_score")),
               "works_per_1k": compare_groups("works_per_1k", by("AI", "works_per_1k"), by("HU", "works_per_1k")),
               "median_works": {c: statistics.median(by(c, "n_works") or [0]) for c in ("AI", "HU")},
               "extraction_suspect": {c: sum(1 for r in rows if r["corpus"] == c and r["extraction_suspect"]) for c in ("AI", "HU")},
               "note": "works_per_1k is the graded observation (lower = thinner grounding); it is not mixed into slop_score"}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"} | {"n_failures": len(fails)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
