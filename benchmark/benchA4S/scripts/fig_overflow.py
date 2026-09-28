"""Does the figure run outside the page's own text area, counted on both benches.

A figure inserted without being scaled to the column overflows the margins the rest of the page
keeps, and the overflow is clipped or crowds the text. The text area is taken from the page's own
body lines rather than from a fixed margin, so a two-column and a one-column layout are read the
same way. Deterministic; no model.
"""
import json, os, sys, statistics as st
import fitz

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
CEN = f"{SLOP}/Artifacts/candidte/_census_0914"
A = f"{B}/../data/Agents4Science"


def text_area(page):
    xs0, xs1 = [], []
    for b in page.get_text("blocks"):
        t = b[4].strip()
        if len(t) > 120:                      # body paragraphs only, not labels inside art
            xs0.append(b[0]); xs1.append(b[2])
    if len(xs0) < 2:
        return None
    return min(xs0), max(xs1)


def overflow(pdf, pno, rect):
    page = fitz.open(pdf)[pno]
    ta = text_area(page)
    if ta is None:
        return None
    left = max(0.0, ta[0] - rect.x0)
    right = max(0.0, rect.x1 - ta[1])
    return (left + right) / max(1.0, ta[1] - ta[0])


def figure_rects(pdf, pno):
    page = fitz.open(pdf)[pno]
    parts = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    parts += [fitz.Rect(im["bbox"]) for im in page.get_image_info()]
    parts = [r for r in parts if r.width > 20 and r.height > 20]
    return parts


def run(rows, label):
    vals = {}
    for corpus, pdf, pno in rows:
        if not os.path.exists(pdf):
            continue
        parts = figure_rects(pdf, pno)
        if not parts:
            continue
        u = parts[0]
        for r in parts[1:]:
            u |= r
        o = overflow(pdf, pno, u)
        if o is not None:
            vals.setdefault(corpus, []).append(o)
    print(f"=== {label}")
    for c, v in vals.items():
        print(f"  {c}: n={len(v)} | median overflow ratio {st.median(v):.3f} | over 5% of text width {sum(1 for x in v if x>0.05)/len(v):.1%}"
              f" | over 15% {sum(1 for x in v if x>0.15)/len(v):.1%}")


cands = json.load(open(f"{B}/figexp/data/candidates.json"))
man = {m["code"]: m for m in json.load(open(f"{B}/figexp/data/manifest.json"))}
pairs = {p["code"]: p for p in json.load(open(f"{A}/pairs_a4s.json"))["pairs"]}
ROOT = os.environ.get("SCISLOP_ROOT", ".")
rows = []
for code, v in cands.items():
    m = man.get(code)
    if m and m.get("ai_png") and v["ai"]:
        rows.append(("AI", os.path.join(ROOT, pairs[code]["ai"]["pdf"]), v["ai"][0]["page"]))
    if m and m.get("hu_png") and v["hu"]:
        rows.append(("HU", f"{B}/figexp/data/hu_pdf/{m['arxiv']}.pdf", v["hu"][0]["page"]))
run(rows, "benchA4S")
