#!/usr/bin/env python3
"""xsec_ref  -  Cross-section reuse of the paper's own objects (Structure plane).

The unit is an OBJECT the paper declares, namely a top-level body section and every labelled
figure, table, equation, algorithm or theorem inside one. An object is reused when some section
other than the one it sits in points to it. The score is the share of objects that are never
reused (slop_definition_revised_0909.md 1.1).

Counting pointers was rejected. A within-section pointer ("Table 1 shows the main results") is
good practice, and a share whose denominator is the pointer count penalises it. The denominator
here is the paper's own objects, so a paper may point at its local table as often as it likes
without moving the score.

Three parsers produce pointer events, and the target of an event is an object.
  K1  reference macros, matched broadly (\\ref, \\cref, \\autoref, \\Secref, \\twosecrefs ...)
  K2  printed section numbers in prose ("Section 4", "Sec. 3.2", "Appendix B")
  K3  deictic and named section references ("the previous section", "in the Method section")
A label declared in the heading zone of a section is an alias of that section, decided by
position rather than by prefix or environment, so a mis-typed label cannot invent a section.

Usage:  python3 measure.py [--limit N] [--only ids]
        -> ../results/{papers.jsonl, objects.jsonl, summary.json}
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import (load_doc, prose_view, sentences, ref_events, tokens, clean_title,
                   REF_MACRO_RE, REF_BLACKLIST, STOP)
from records import instance, write_jsonl, compare_groups, confound_audit, rate, describe, slop_score

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "xsec_ref_v3"

# --------------------------------------------------------------------------- objects

HEADING_RE = re.compile(r"\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def section_alias_labels(doc) -> dict:
    """label -> (top-level section index, heading level), for labels declared in a heading zone.

    A label is an alias of a heading when the text between the end of the nearest preceding
    heading and the label contains no other label and no \\begin{...}. Position decides, so a
    label named `prop:geometry` inside a theorem is never read as a heading (the kind heuristic
    used to do exactly that) and a label named `sec:x` inside a figure is never read as one.
    A level-1 alias names the section itself; a level-2 or level-3 alias names a sub-section,
    which is a separate object living inside that section.
    """
    out = {}
    heads = [(m, len(re.match(r"\\(sub)*", m.group(0)).group(0)) // 3 + 1)
             for m in HEADING_RE.finditer(doc.tex) if doc.doc_start <= m.start() < doc.doc_end]
    for m in LABEL_RE.finditer(doc.tex):
        if not (doc.doc_start <= m.start() < doc.doc_end):
            continue
        prev = [h for h in heads if h[0].end() <= m.start()]
        nxt = [h for h in heads if h[0].start() >= m.end()]
        cands = []
        if prev:
            head, level = prev[-1]
            gap = doc.tex[head.end():m.start()]
            if not ("\\begin{" in gap or "\\label{" in gap or len(gap) > 120):
                cands.append((len(gap), head.start(), level))
        if nxt:
            head, level = nxt[0]
            gap = doc.tex[m.end():head.start()]
            # a label written just before its own heading belongs to that heading, not to the section above
            if gap.strip() == "":
                cands.append((len(gap), head.start(), level))
        if not cands:
            continue
        _, hstart, level = min(cands)
        idx = doc.top_of(hstart)
        if idx is not None:
            out[m.group(1).strip()] = (idx, level)
    return out


def build_objects(doc):
    """Objects of the paper.

    Every top-level body section is an object with id ('S', idx). Every other label whose
    declaration sits in a body section is an object with id ('L', label). A label written on a
    section heading is another name for that section, and a label written on a sub-section
    heading is an object living inside its parent section. Appendix material is not an object;
    pointers into it are counted separately.
    """
    body = doc.body_sections()
    body_idx = {s.idx for s in body}
    alias = section_alias_labels(doc)
    objects, label_to_obj = {}, {}
    for s in body:
        objects[("S", s.idx)] = {"kind": "section", "home": s.idx, "title": s.title, "role": s.role,
                                 "offset": s.start, "n_labels": 0}
    for lab, info in doc.labels.items():
        home = info["section"]
        if home is None or home not in body_idx:
            continue
        if lab in alias and alias[lab][0] in body_idx and alias[lab][1] == 1:
            # a label on the section heading itself is another name for the section object
            label_to_obj[lab] = ("S", alias[lab][0])
            objects[("S", alias[lab][0])]["n_labels"] += 1
            continue
        kind = "subsection" if lab in alias else (info["kind"] if info["kind"] != "section" else "other")
        oid = ("L", lab)
        objects[oid] = {"kind": kind, "home": home, "title": lab, "role": None,
                        "offset": info["offset"], "n_labels": 1}
        label_to_obj[lab] = oid
    return objects, label_to_obj, body_idx


# --------------------------------------------------------------------------- parsers

TEXT_SECNUM_RE = re.compile(
    r"\b(?:Sections?|Secs?\.|Appendices|Appendix|App\.|§)\s*~?\s*"
    r"([A-Z]?\d+(?:\.\d+)*|[A-Z])(?:\s*(?:,|and|&|-|–|to)\s*~?\s*([A-Z]?\d+(?:\.\d+)*|[A-Z]))?")
DEICTIC_RE = re.compile(r"\bthe\s+(previous|preceding|prior|last|next|following|subsequent)\s+(?:sub)?section\b", re.I)
NAMED_RE = re.compile(r"\b(?:in|see|of|from|described in|discussed in|presented in|introduced in|defined in)\s+"
                      r"(?:the\s+)?([A-Za-z][A-Za-z\- ]{2,40}?)\s+(?:sub)?section\b", re.I)
APPENDIX_WORD_RE = re.compile(r"\b(?:in|see|to|defer\w*\s+to|provided in|reported in)\s+(?:the\s+)?appendix\b", re.I)
ROLE_WORDS = {"introduction": "Intro", "intro": "Intro", "related work": "Related", "background": "Related",
              "method": "Method", "methods": "Method", "methodology": "Method", "approach": "Method",
              "experiment": "Experiments", "experiments": "Experiments", "experimental": "Experiments",
              "results": "Experiments", "evaluation": "Experiments", "discussion": "Conclusion",
              "conclusion": "Conclusion", "conclusions": "Conclusion"}


def section_numbers(doc):
    """Printed number of each top-level section. Body sections are 1..k, starred ones are unnumbered,
    appendix sections are A, B, C. Returns number -> section index."""
    out, k, letter = {}, 0, ord("A")
    for s in doc.top_sections:
        star = re.match(r"\\section\*", doc.tex[s.start:s.start + 12]) is not None
        if s.in_appendix or s.role == "Appendix":
            if not star:
                out[chr(letter)] = s.idx; letter += 1
        elif not star:
            k += 1; out[str(k)] = s.idx
    return out


def title_index(doc, body_idx):
    """Content tokens of each body section title, and role -> section index, for named references."""
    ttoks, byrole = {}, {}
    for s in doc.body_sections():
        ttoks[s.idx] = {t.lower() for t in tokens(clean_title(s.title)) if t.lower() not in STOP and len(t) > 2}
        byrole.setdefault(s.role, []).append(s.idx)
    return ttoks, {r: v[0] for r, v in byrole.items() if len(v) == 1}


def collect_events(doc, objects, label_to_obj, body_idx):
    """Pointer events from the three parsers.

    Each event is {parser, src, target, resolved, offset, kind}. `target` is an object id, the
    string 'appendix' for a pointer into the appendix, or None when the parser could not resolve
    it (which lowers coverage and nothing else)."""
    ev = []
    # K1 macros
    for e in ref_events(doc, include_appendix=True):
        if e["src_section"] is None or e["src_section"] not in body_idx or e["src_in_appendix"]:
            continue
        rec = {"parser": "K1", "src": e["src_section"], "offset": e["offset"], "macro": e["macro"], "label": e["label"]}
        if e["dst_kind"] == "undefined":
            ev.append({**rec, "target": None, "resolved": False, "why": "label never declared"}); continue
        if e["dst_in_appendix"] or (e["dst_section"] is not None and e["dst_section"] not in body_idx):
            ev.append({**rec, "target": "appendix", "resolved": True}); continue
        oid = label_to_obj.get(e["label"])
        if oid is None:
            ev.append({**rec, "target": None, "resolved": False, "why": "label declared outside the body"}); continue
        ev.append({**rec, "target": oid, "resolved": True})
    # K2 literal section numbers, K3 deictic and named
    nums = section_numbers(doc)
    ttoks, byrole = title_index(doc, body_idx)
    order = [s.idx for s in doc.body_sections()]
    mt = prose_view(doc.tex[doc.doc_start:doc.body_end()], base=doc.doc_start)
    for s in sentences(mt):
        src = doc.top_of(s["tex_span"][0])
        if src is None or src not in body_idx:
            continue
        base = {"src": src, "offset": s["tex_span"][0]}
        for m in TEXT_SECNUM_RE.finditer(s["text"]):
            for g in (m.group(1), m.group(2)):
                if not g:
                    continue
                top = g.split(".")[0]
                idx = nums.get(top)
                rec = {**base, "parser": "K2", "text": m.group(0)[:40]}
                if idx is None:
                    ev.append({**rec, "target": None, "resolved": False, "why": "printed number not found"})
                elif idx not in body_idx:
                    ev.append({**rec, "target": "appendix", "resolved": True})
                else:
                    ev.append({**rec, "target": ("S", idx), "resolved": True})
        for m in DEICTIC_RE.finditer(s["text"]):
            step = -1 if m.group(1).lower() in ("previous", "preceding", "prior", "last") else 1
            pos = order.index(src) + step if src in order else -1
            rec = {**base, "parser": "K3", "text": m.group(0)[:40]}
            if 0 <= pos < len(order):
                ev.append({**rec, "target": ("S", order[pos]), "resolved": True})
            else:
                ev.append({**rec, "target": None, "resolved": False, "why": "no adjacent section"})
        for m in NAMED_RE.finditer(s["text"]):
            name = m.group(1).lower().strip()
            rec = {**base, "parser": "K3", "text": m.group(0)[:40]}
            idx = byrole.get(ROLE_WORDS.get(name, ""))
            if idx is None:
                nt = {t.lower() for t in tokens(name) if t.lower() not in STOP and len(t) > 2}
                hits = [i for i, tt in ttoks.items() if nt and nt <= tt] if nt else []
                idx = hits[0] if len(hits) == 1 else None
            if idx is None:
                ev.append({**rec, "target": None, "resolved": False, "why": "section name not matched"})
            else:
                ev.append({**rec, "target": ("S", idx), "resolved": True})
        for m in APPENDIX_WORD_RE.finditer(s["text"]):
            ev.append({**base, "parser": "K3", "text": m.group(0)[:40], "target": "appendix", "resolved": True})
    return ev


# --------------------------------------------------------------------------- roadmap

def roadmap_events(doc, events, objects, body_idx):
    """Structural definition, no lexicon. Inside the first body section, take the pointer events
    whose target is a later section object, in order of occurrence. The longest run that visits
    strictly increasing, distinct sections and has at least two members is the roadmap."""
    body = doc.body_sections()
    if not body:
        return set()
    intro = body[0].idx
    seq = [e for e in sorted(events, key=lambda x: x["offset"])
           if e["src"] == intro and isinstance(e["target"], tuple) and e["target"][0] == "S" and e["target"][1] > intro]
    best, cur = [], []
    for e in seq:
        if cur and e["target"][1] <= cur[-1]["target"][1]:
            if len(cur) > len(best):
                best = cur
            cur = []
        cur.append(e)
    if len(cur) > len(best):
        best = cur
    return {id(e) for e in best} if len(best) >= 2 else set()


# --------------------------------------------------------------------------- measurement

def measure(paper: dict):
    doc = load_doc(paper)
    objects, label_to_obj, body_idx = build_objects(doc)
    events = collect_events(doc, objects, label_to_obj, body_idx)
    road = roadmap_events(doc, events, objects, body_idx)

    reach, reach_road, local = defaultdict(set), defaultdict(set), Counter()
    n_app = 0
    for e in events:
        if e["target"] == "appendix":
            n_app += 1; continue
        if not isinstance(e["target"], tuple):
            continue
        home = objects[e["target"]]["home"]
        if e["src"] == home:
            local[e["target"]] += 1
        else:
            reach_road[e["target"]].add(e["src"])
            if id(e) not in road:
                reach[e["target"]].add(e["src"])

    rows = []
    for oid, o in objects.items():
        rows.append({"paper": paper["id"], "corpus": paper["corpus"], "object": f"{oid[0]}:{oid[1]}",
                     "kind": o["kind"], "home_section": o["home"], "home_role": doc.top_sections[o["home"]].role,
                     "title": o["title"][:80], "n_local_pointers": local[oid],
                     "reach": len(reach[oid]), "reach_incl_roadmap": len(reach_road[oid]),
                     "reached_from": sorted(reach[oid]), "reused": len(reach[oid]) > 0,
                     "referenced_at_all": bool(local[oid] or reach_road[oid])})

    n_obj = len(rows)
    n_reused = sum(1 for r in rows if r["reused"])
    n_used = sum(1 for r in rows if r["referenced_at_all"])
    resolved = sum(1 for e in events if e["resolved"])
    bykind = defaultdict(lambda: [0, 0])
    for r in rows:
        bykind[r["kind"]][1] += 1
        bykind[r["kind"]][0] += int(r["reused"])
    byrole = defaultdict(lambda: [0, 0])
    for r in rows:
        byrole[r["home_role"]][1] += 1
        byrole[r["home_role"]][0] += int(r["reused"])
    back = sum(1 for r in rows for s in r["reached_from"] if s > r["home_section"])
    fwd = sum(1 for r in rows for s in r["reached_from"] if s < r["home_section"])

    row = {
        "corpus": paper["corpus"], "id": paper["id"],
        "source_incomplete": bool(doc.report["missing"]), "n_missing_inputs": len(doc.report["missing"]),
        "n_words_body": doc.n_words_body, "n_sections_body": len(body_idx), "n_objects": n_obj,
        "n_objects_by_kind": {k: v[1] for k, v in bykind.items()},
        "n_reused": n_reused, "n_referenced_at_all": n_used,
        "reuse_rate": rate(n_reused, n_obj), "referenced_rate": rate(n_used, n_obj),
        "mean_reach": round(sum(r["reach"] for r in rows) / n_obj, 4) if n_obj else None,
        "max_reach": max([r["reach"] for r in rows], default=0),
        "reuse_rate_by_kind": {k: rate(v[0], v[1]) for k, v in bykind.items()},
        "reuse_rate_by_home_role": {k: rate(v[0], v[1]) for k, v in byrole.items()},
        "backward_links": back, "forward_links": fwd,
        "n_roadmap_events": len(road),
        "reuse_rate_incl_roadmap": rate(sum(1 for r in rows if r["reach_incl_roadmap"] > 0), n_obj),
        "appendix_pointers": n_app,
        "n_events": len(events), "n_events_resolved": resolved,
        "events_by_parser": dict(Counter(e["parser"] for e in events)),
        "unresolved_reasons": dict(Counter(e.get("why") for e in events if not e["resolved"])),
        "has_reuse": n_reused > 0,
        # slop level (SLOP_SCORE.md): objects never reused outside their home section / objects
        **slop_score(n_obj - n_reused, n_obj), "coverage": rate(resolved, len(events)),
    }
    return row, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    papers = ai_papers() + hu_papers()
    if a.only:
        ids = set(a.only.split(",")); papers = [p for p in papers if p["id"] in ids]
    if a.limit:
        papers = papers[:a.limit]
    rows, objs, fails = [], [], []
    for p in papers:
        try:
            r, o = measure(p); rows.append(r); objs.extend(o)
        except Exception as ex:
            fails.append({"id": p["id"], "error": repr(ex)[:300]})
    write_jsonl(os.path.join(RESULTS, "papers.jsonl"), rows)
    write_jsonl(os.path.join(RESULTS, "objects.jsonl"), objs)

    ai = [r for r in rows if r["corpus"] == "AI"]
    hu = [r for r in rows if r["corpus"] == "HU" and not r["source_incomplete"]]
    hu_all = [r for r in rows if r["corpus"] == "HU"]
    summary = {"checker": CHECKER, "n_ai": len(ai), "n_hu_complete": len(hu),
               "n_hu_incomplete_source": len(hu_all) - len(hu), "failures": fails,
               "unit": "object = a body section, or a labelled figure/table/equation/algorithm/theorem inside one",
               "score": "1 - reused objects / objects; a within-section pointer never moves it",
               "metrics": {}}
    for m in ["slop_score", "reuse_rate", "referenced_rate", "mean_reach", "reuse_rate_incl_roadmap",
              "coverage", "n_objects", "n_sections_body", "n_words_body", "appendix_pointers",
              "backward_links", "forward_links"]:
        summary["metrics"][m] = compare_groups(m, [r[m] for r in ai if r[m] is not None],
                                               [r[m] for r in hu if r[m] is not None])
    kinds = sorted({k for r in rows for k in r["reuse_rate_by_kind"]})
    summary["reuse_rate_by_kind"] = {c: {k: rate(sum(o["reused"] for o in objs if o["corpus"] == c and o["kind"] == k),
                                                 sum(1 for o in objs if o["corpus"] == c and o["kind"] == k)) for k in kinds}
                                     for c in ("AI", "HU")}
    summary["has_reuse_share"] = {"AI": rate(sum(1 for r in ai if r["has_reuse"]), len(ai)),
                                  "HU": rate(sum(1 for r in hu if r["has_reuse"]), len(hu))}
    summary["parser_contribution"] = {c: dict(sum((Counter(r["events_by_parser"]) for r in rows if r["corpus"] == c), Counter()))
                                      for c in ("AI", "HU")}
    summary["unresolved_reasons"] = {c: dict(sum((Counter(r["unresolved_reasons"]) for r in rows if r["corpus"] == c), Counter()))
                                     for c in ("AI", "HU")}
    strata = {}
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 10 ** 6)]:
        strata[f"objects_{lo}_{hi}"] = {c: describe([r["slop_score"] for r in g if lo <= r["n_objects"] < hi])
                                        for c, g in (("AI", ai), ("HU", hu))}
    summary["strata_by_object_count"] = strata
    pos = [r for r in ai if not r["has_reuse"]]; neg = [r for r in ai if r["has_reuse"]]
    summary["confound_audit_AI_no_reuse"] = {f: confound_audit([r[f] for r in pos], [r[f] for r in neg], f)
                                             for f in ["n_objects", "n_words_body", "n_sections_body", "n_events"]}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)

    print(json.dumps({k: summary[k] for k in ["n_ai", "n_hu_complete", "n_hu_incomplete_source",
                                              "has_reuse_share", "reuse_rate_by_kind", "parser_contribution"]}, indent=1))
    for m in ["slop_score", "reuse_rate", "referenced_rate", "mean_reach", "coverage", "n_objects"]:
        s = summary["metrics"][m]
        print(f"  {m:22} AI mean {s['AI'].get('mean')} med {s['AI'].get('median')} | HU mean {s['HU'].get('mean')} med {s['HU'].get('median')} | cliff {s['cliff_delta_AI_minus_HU']}")
    print("  confound (AI, no reuse vs reuse):", {k: v["auroc"] for k, v in summary["confound_audit_AI_no_reuse"].items()})
    if fails:
        print("FAILURES", fails)


if __name__ == "__main__":
    main()
