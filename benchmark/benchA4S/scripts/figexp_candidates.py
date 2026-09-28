"""fig_exposition inputs for benchA4S, step 1 of 2: propose method-figure candidates.

Both sides are localised the same way because A4S papers have no FARS framework_overview image.
Every captioned figure becomes a candidate and the crop is the union of the vector drawings and
raster images that sit above the caption inside its column, so body text above the figure is never
swept in and a caption with no graphics above it never reaches the measurer. Which candidate is the
method figure is left to scripts/figexp_gate.py, because a caption keyword decides that badly in
both directions, and one rule applied by one reader keeps the two corpora read the same way.

Writes figexp/data/{candidates.json, cand/<side>_<id>__f<n>.png}.
"""
import json, os, re, sys, time, urllib.request

import fitz

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S"
D = f"{B}/figexp/data"
CAND = f"{D}/cand"
PAIRS = os.environ.get("A4S_PAIRS", f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science/pairs_a4s.json")

# A caption, not a sentence that mentions a figure, so the number must be followed by its separator.
CAPTION = re.compile(r"^\s*(Figure|Fig\.?)\s*(\d{1,2})\s*[:.]\s")


def caption_blocks(page):
    out = []
    for b in page.get_text("blocks"):
        txt = b[4].replace("\n", " ").strip()
        m = CAPTION.match(txt)
        if m:
            out.append((fitz.Rect(b[:4]), txt, int(m.group(2))))
    return out


def figure_clip(page, cap_rect):
    """Union of graphics above the caption inside its column, or None."""
    band_lo, band_hi = cap_rect.x0 - 12, cap_rect.x1 + 12
    parts = []
    for d in page.get_drawings():
        parts.append(fitz.Rect(d["rect"]))
    for im in page.get_image_info():
        parts.append(fitz.Rect(im["bbox"]))
    keep = []
    for r in parts:
        if r.y1 > cap_rect.y0 + 2 or r.y0 < cap_rect.y0 - 520:
            continue
        if r.x1 < band_lo or r.x0 > band_hi:
            continue
        if r.width < 3 and r.height < 3:
            continue
        keep.append(r)
    if not keep:
        return None
    u = keep[0]
    for r in keep[1:]:
        u |= r
    u = fitz.Rect(max(page.rect.x0, u.x0 - 8), max(page.rect.y0, u.y0 - 8),
                  min(page.rect.x1, u.x1 + 8), min(cap_rect.y0 - 1, u.y1 + 8))
    if u.height < 55 or u.width < 90 or u.get_area() < 9000:
        return None
    return u


def candidates(pdf, side, ident, limit=8):
    try:
        doc = fitz.open(pdf)
    except Exception:
        return []
    out, seen = [], set()
    for pno in range(min(len(doc), 25)):
        page = doc[pno]
        for cap, txt, fno in caption_blocks(page):
            clip = figure_clip(page, cap)
            if clip is None or (pno, round(clip.x0), round(clip.y0)) in seen:
                continue
            seen.add((pno, round(clip.x0), round(clip.y0)))
            png = f"{CAND}/{side}_{ident}__f{len(out)}.png"
            page.get_pixmap(clip=clip, dpi=150).save(png)
            out.append({"png": png, "page": pno, "fig_no": fno, "caption": txt[:400]})
            if len(out) >= limit:
                return out
    return out


def arxiv_pdf(ax, out):
    if os.path.isfile(out) and os.path.getsize(out) > 20000:
        return True
    for host in ("arxiv.org", "export.arxiv.org"):
        try:
            b = urllib.request.urlopen(urllib.request.Request(
                f"https://{host}/pdf/{ax}",
                headers={"User-Agent": "scislopbench-a4s/0.4 (user@example.org)"}), timeout=90).read()
        except Exception:
            continue
        if b[:4] == b"%PDF":
            open(out, "wb").write(b)
            time.sleep(3)
            return True
    return False


def main():
    os.makedirs(CAND, exist_ok=True)
    os.makedirs(f"{D}/hu_pdf", exist_ok=True)
    pairs = json.load(open(PAIRS))["pairs"]
    pairs = [p for p in pairs if p.get("human") or p.get("human_primary")]
    rec = {}
    for i, p in enumerate(pairs):
        hu = p.get("human") or p["human_primary"]
        code, ax = p["code"], hu["arxiv"]
        ai = candidates(os.path.join(ROOT, p["ai"]["pdf"]), "AI", code)
        hp = f"{D}/hu_pdf/{ax}.pdf"
        hucand = candidates(hp, "HU", ax) if arxiv_pdf(ax, hp) else []
        rec[code] = {"arxiv": ax, "ai": ai, "hu": hucand}
        print(f"  [{i+1}/{len(pairs)}] {code} AI {len(ai)} | HU {ax} {len(hucand)}", flush=True)
    json.dump(rec, open(f"{D}/candidates.json", "w"), indent=1)
    print(f"papers {len(rec)} | AI candidates {sum(len(v['ai']) for v in rec.values())} | "
          f"HU candidates {sum(len(v['hu']) for v in rec.values())}")


if __name__ == "__main__":
    main()
