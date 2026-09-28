"""fig_exposition inputs for benchA4S: a manifest, the AI method figure, the human method figure.

The item's usual AI input is the FARS pipeline's framework_overview image, which A4S papers do
not have, so both sides are localised the same way here: find the first caption that passes the
method-figure gate, crop the region above it, render at 150 dpi. The human PDF is fetched from
arXiv because the pair set carries only its LaTeX.

Writes figexp/data/{manifest.json, hu_figure.json, crops/{AI_<code>,HU_<arxiv>}.png}.
"""
import json, os, re, subprocess, sys, urllib.request

import fitz

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S"
D = f"{B}/figexp/data"
CROPS = f"{D}/crops"
GATE = re.compile(r"\b(overview|framework|pipeline|architecture|schematic|workflow|illustrat\w*|"
                  r"diagram|our (?:method|approach|framework|model|system)|"
                  r"proposed (?:method|framework|approach|model)|we propose|process of|procedure)\b", re.I)
RESULTY = re.compile(r"\b(accuracy|performance|results?|comparison of|curves?|trade-?off|evaluation|"
                     r"win rate|scores?|perplexity|loss|speed|throughput|latency|ablation|"
                     r"distribution of|correlation|heat ?map)\b", re.I)


def method_figure(pdf):
    """(page index, clip rect) of the first caption that reads like a method figure."""
    doc = fitz.open(pdf)
    for pno in range(len(doc)):
        page = doc[pno]
        for m in re.finditer(r"(Figure|Fig\.)\s*\d{1,2}\s*[:.]", page.get_text()):
            cap_start = m.group(0)
            rects = page.search_for(cap_start)
            if not rects:
                continue
            r = rects[0]
            tail = page.get_text()[m.end():m.end() + 320]
            if RESULTY.search(tail) and not GATE.search(tail):
                continue
            if not GATE.search(tail):
                continue
            top = 36
            clip = fitz.Rect(max(0, r.x0 - 20), top, min(page.rect.x1, r.x1 + 320), r.y0 - 2)
            if clip.height < 60 or clip.width < 80:
                continue
            return pno, clip
    return None, None


def crop(pdf, out):
    pno, clip = method_figure(pdf)
    if pno is None:
        return False
    page = fitz.open(pdf)[pno]
    page.get_pixmap(clip=clip, dpi=150).save(out)
    return True


def arxiv_pdf(ax, out):
    if os.path.isfile(out) and os.path.getsize(out) > 20000:
        return True
    try:
        b = urllib.request.urlopen(urllib.request.Request(
            f"https://arxiv.org/pdf/{ax}", headers={"User-Agent": "scislopbench-a4s/0.3 (user@example.org)"}),
            timeout=90).read()
    except Exception:
        try:
            b = urllib.request.urlopen(urllib.request.Request(
                f"https://export.arxiv.org/pdf/{ax}", headers={"User-Agent": "scislopbench-a4s/0.3"}),
                timeout=90).read()
        except Exception:
            return False
    if b[:4] != b"%PDF":
        return False
    open(out, "wb").write(b)
    return True


def main():
    os.makedirs(CROPS, exist_ok=True)
    os.makedirs(f"{D}/hu_pdf", exist_ok=True)
    pairs = json.load(open(f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science/pairs_a4s_v7.json"))["pairs"]
    man, hu_fig = [], {}
    import time
    for p in pairs:
        code, ax = p["code"], p["human"]["arxiv"]
        ai_pdf = os.path.join(ROOT, p["ai"]["pdf"])
        ai_png = f"{CROPS}/AI_{code}.png"
        ai_ok = crop(ai_pdf, ai_png)
        hp = f"{D}/hu_pdf/{ax}.pdf"
        hu_ok = False
        if arxiv_pdf(ax, hp):
            time.sleep(3)
            hu_ok = crop(hp, f"{CROPS}/HU_{ax}.png")
        hu_fig[ax] = "found" if hu_ok else "none"
        man.append({"code": code, "arxiv": ax, "ai_png": ai_png if ai_ok else None})
        print(f"  {code} AI {'o' if ai_ok else 'x'} | HU {ax} {'o' if hu_ok else 'x'}", flush=True)
    json.dump(man, open(f"{D}/manifest.json", "w"), indent=1)
    json.dump(hu_fig, open(f"{D}/hu_figure.json", "w"), indent=1)
    print(f"manifest {len(man)} | AI figures {sum(1 for m in man if m['ai_png'])} | "
          f"HU figures {sum(1 for v in hu_fig.values() if v != 'none')}")


if __name__ == "__main__":
    main()
