"""Convert an A4S submission PDF into the TeX-shaped document the slop loader expects.

FARS papers arrive as LaTeX, A4S papers as PDF, and every checker reads `main_tex` through
views.expand_tex. Rather than write a second reader, the PDF is re-expressed in the small subset
of LaTeX the loader actually parses, so both sides go through one code path:

  \\section{...}      from the headings the PDF actually prints, found by font weight and size
  \\subsection{...}   from a heading whose number has a second component, "4.2 Ablations"
  \\label{...}        for each float caption "Figure 3:" -> \\label{fig:3}
  \\cite{...}         for each in-text citation marker, numeric [12] or author-year (Name, 2020)
  \\ref{...}          for each in-text pointer "Figure 3" / "Table 2" / "Section 4"
  figure/table env    for each caption, so float counts match the TeX side
  tabular             the cells of a result table, rebuilt from the word positions under its
                      caption, because a table environment with no rows is not a table to any
                      checker that reads cells

Section structure comes from the rendered page and not from meta.json, whose section table is a
fixed set of canonical buckets (abstract, introduction, related, method, ...) and therefore gives
every paper the same six to nine sections no matter what it wrote. Any measure that reads the
section partition, cross-section reference or repetition across sections, would be reading that
bucketing rather than the paper. meta.json is the fallback for a PDF whose headings cannot be
found at all.

What this cannot recover is stated in the header of every file it writes. A pointer the authors
never wrote is not invented, and a citation the PDF renders as a bare number is recovered only
when the reference list resolves it, so citation keys are reference indices and not bib keys.
"""
import json, os, re, sys

import fitz

ROOT = os.environ.get("SCISLOP_ROOT", ".")
A4S = f"{ROOT}/Agents4Science/2025_full"
OUT = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S/ai_tex"

LINENO = re.compile(r"^\s*\d{1,4}\s*$")
CAPTION = re.compile(r"\b(Figure|Fig\.|Table|Algorithm)\s+(\d{1,2})\s*[:.]\s*", re.I)
POINTER = re.compile(r"\b(Figure|Fig\.|Table|Section|Sec\.|Equation|Eq\.|Algorithm|Appendix)\s*~?\s*(\d{1,2}|[A-D])\b")
NUMCITE = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
# the corpus uses four styles: [12] / [Author et al., 2023, Other, 2024] / Author et al. [2021] /
# (Author et al., 2020). All four are turned into \cite so the citing-sentence unit exists.
BRACKET_AY = re.compile(r"\[((?:[A-Z][^\[\]]{2,160}?)(?:19|20)\d\d[a-z]?)\]")
CITET = re.compile(r"\b([A-Z][A-Za-zÀ-ÿ'`\-]+(?:\s+(?:et\s+al\.?|and|&)\s*[A-Z]?[A-Za-zÀ-ÿ'`\-]*)*)\s*\[((?:19|20)\d\d[a-z]?(?:\s*,\s*(?:19|20)\d\d[a-z]?)*)\]")
PAREN_AY = re.compile(r"\(((?:[A-Z][A-Za-zÀ-ÿ'`\-]+(?:\s+(?:et al\.?|and|&|[A-Z][A-Za-zÀ-ÿ'`\-]+))*,?\s+(?:19|20)\d\d[a-z]?)(?:\s*;\s*[^()]{3,80})?)\)")
KIND = {"figure": "fig", "fig.": "fig", "table": "tab", "algorithm": "alg",
        "section": "sec", "sec.": "sec", "equation": "eq", "appendix": "app"}
TABLE_CAP = re.compile(r"^\s*Table\s*(\d{1,2})\s*[:.]\s")
NUMCH = re.compile(r"\d")
BODY_SKIP = {"abstract", "references", "appendix", "ai_disclosure", "acknowledgments",
             "acknowledgements", "checklist", "broader_impacts", "ethics"}
# the same non-body matter, matched inside a printed heading rather than as a bucket name
BODY_SKIP_WORDS = ("reference", "bibliograph", "acknowledg", "checklist", "broader impact",
                   "ai involvement", "disclosure", "ethics statement", "appendi",
                   "supplementary", "reproducibility statement")


WORD = re.compile(r"[A-Za-z][A-Za-z\u00c0-\u00ff'`-]*")
BREAK_HYPH = re.compile(r"([A-Za-z\u00c0-\u00ff]{2,})-[ \t]*\n[ \t]*([a-z\u00e0-\u00ff][A-Za-z\u00c0-\u00ff'`-]*)")


def vocabulary(secs):
    """Words the PDF writes without a line break, used to decide what a broken hyphen was."""
    v = set()
    for t in secs.values():
        for line in str(t).split("\n"):
            for w in WORD.findall(line):
                v.add(w.lower())
    return v


def dehyphenate(t, vocab):
    """A hyphen at a line end is either the break of one word or a compound TeX chose to break at.
    Rejoining both the same way would invent words, so the document's own spelling decides."""
    def rep(m):
        a, b = m.group(1), m.group(2)
        joined, kept = (a + b).lower(), (a + "-" + b).lower()
        if joined in vocab and kept not in vocab:
            return a + b
        if kept in vocab and joined not in vocab:
            return a + "-" + b
        return a + b
    return BREAK_HYPH.sub(rep, t)


def _lines(page, rect, band):
    ws = [w for w in page.get_text("words")
          if w[1] >= rect.y0 - 1 and w[3] <= rect.y1 + 1 and w[2] > band[0] - 6 and w[0] < band[1] + 6]
    ws.sort(key=lambda w: (round(w[1], 1), w[0]))
    out, cur, y = [], [], None
    for w in ws:
        if y is None or abs(w[1] - y) <= 3:
            cur.append(w)
        else:
            out.append(cur)
            cur = [w]
        y = w[1] if y is None or abs(w[1] - y) > 3 else y
    if cur:
        out.append(cur)
    return [sorted(l, key=lambda w: w[0]) for l in out]


def _cells(line, gap=7.0):
    cs, cur = [], [line[0]]
    for a, b in zip(line, line[1:]):
        if b[0] - a[2] > gap:
            cs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    cs.append(cur)
    return [" ".join(w[4] for w in c) for c in cs]


def _table_rows(page, cr):
    """Cells of the table that belongs to this caption, from the column band under or over it."""
    band = (cr.x0, cr.x1)
    for rect in (fitz.Rect(band[0] - 10, cr.y1, band[1] + 10, min(page.rect.y1, cr.y1 + 330)),
                 fitz.Rect(band[0] - 10, max(page.rect.y0, cr.y0 - 330), band[1] + 10, cr.y0)):
        rows = []
        for line in _lines(page, rect, band):
            c = _cells(line)
            wide = len(line) > 16 and min((b[0] - a[2]) for a, b in zip(line, line[1:]) or [(0, (0, 0, 99))]) < 9
            if len(c) < 2 or wide:
                if rows:
                    break
                continue
            rows.append(c)
        if len(rows) >= 3 and sum(1 for r in rows if any(NUMCH.search(x) for x in r)) >= 2:
            return rows
    return None


NUMHEAD = re.compile(r"^(\d{1,2})((?:\.\d{1,2}){0,2})\.?\s+\S")
FLOATCAP = re.compile(r"^(Figure|Fig\.|Table|Algorithm|Listing)\b", re.I)
FRONT = re.compile(r"^(abstract)\b", re.I)
# Pseudocode sets its keywords bold on their own short line, which is the heading signature exactly.
PSEUDO = re.compile(r"^(end\s+(for|if|while|procedure|function)|else|elif|"
                    r"for|while|repeat|until|return|require|ensure|input|output|initialize|"
                    r"procedure|function|do)\b[:\s]*$", re.I)
STOP_AT = re.compile(r"^(references|bibliography)\b", re.I)


def _body_face(doc):
    from collections import Counter
    size, font = Counter(), Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for sp in l.get("spans", []):
                    if len(sp["text"].strip()) > 40:
                        size[round(sp["size"], 1)] += len(sp["text"])
                        font[sp["font"]] += len(sp["text"])
    return (size.most_common(1)[0][0] if size else 10.0,
            font.most_common(1)[0][0] if font else "")


def _line_text(spans):
    """One line's text. Each word is its own span with no trailing space, so a span boundary that
    has horizontal gap is a word boundary; joining the spans directly runs the line into one token
    and every sentence and word measure downstream reads that."""
    out = []
    for i, sp in enumerate(spans):
        if i and spans[i]["bbox"][0] - spans[i - 1]["bbox"][2] > 0.12 * max(1.0, sp["size"]):
            if out and not out[-1].endswith(" ") and not sp["text"].startswith(" "):
                out.append(" ")
        out.append(sp["text"])
    return "".join(out)


def _bold(spans):
    return all("bold" in sp["font"].lower() or "-medi" in sp["font"].lower() or (sp["flags"] & 16)
               for sp in spans)


def sections_from_pdf(pdf):
    """[(level, title, text)] in reading order, from the headings the page actually prints.

    The A4S template sets a subsection in the body face, bold but no larger, and PyMuPDF puts it in
    the same block as the paragraph under it, so neither size nor block structure finds it. What
    separates it from bold emphasis inside a paragraph is that the line stops well short of the
    column and the line under it is not bold. Checked against the eighteen submissions that also
    shipped their LaTeX, this recovers a median of 20 body headings where the source has 21.

    Everything before the abstract is the title block and is dropped, and the first reference
    heading ends the document.
    """
    try:
        doc = fitz.open(pdf)
    except Exception:
        return []
    bs, _bf = _body_face(doc)
    secs, cur, started = [], None, False
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            lines = [l for l in b.get("lines", [])
                     if [sp for sp in l.get("spans", []) if sp["text"].strip()]]
            if not lines:
                continue
            col = max(l["bbox"][2] for l in lines) - min(l["bbox"][0] for l in lines) or 1.0
            for li, l in enumerate(lines):
                spans = [sp for sp in l.get("spans", []) if sp["text"].strip()]
                txt = _line_text(spans).strip()
                if not (2 < len(txt) <= 90) or FLOATCAP.match(txt):
                    if started and cur is not None and txt:
                        cur[2].append(txt)
                    continue
                size = max(sp["size"] for sp in spans)
                bold, big = _bold(spans), max(sp["size"] for sp in spans) >= bs + 0.6
                m = NUMHEAD.match(txt)
                short = (l["bbox"][2] - l["bbox"][0]) / col <= 0.75
                nxt = lines[li + 1] if li + 1 < len(lines) else None
                plain_next = nxt is None or not _bold([sp for sp in nxt.get("spans", [])
                                                       if sp["text"].strip()])
                head = ((m and (bold or big)) or (bold and big)
                        or (bold and short and plain_next and not txt.endswith((".", ",", ";"))))
                if head and PSEUDO.match(txt):
                    head = False
                if head:
                    if STOP_AT.match(txt):
                        return secs
                    if FRONT.match(txt):
                        started = True
                    if not started:
                        continue
                    lvl = 1 + (m.group(2).count(".") if m else (0 if big else 1))
                    # a heading whose number and title were laid out as two lines
                    if m and not txt[m.end() - 1:].strip(" .0123456789"):
                        nxt_txt = ("".join(sp["text"] for sp in nxt.get("spans", [])).strip()
                                   if nxt is not None else "")
                        if 0 < len(nxt_txt) <= 60:
                            txt = f"{txt} {nxt_txt}"
                    cur = [min(3, lvl), txt, []]
                    secs.append(cur)
                elif started and cur is not None:
                    cur[2].append(txt)
    return secs


def pdf_tables(pdf):
    """{table number: rows}. A caption whose cells cannot be read keeps its environment empty."""
    try:
        doc = fitz.open(pdf)
    except Exception:
        return {}
    out = {}
    for page in doc:
        for b in page.get_text("blocks"):
            t = b[4].replace("\n", " ").strip()
            m = TABLE_CAP.match(t)
            if not m or int(m.group(1)) in out:
                continue
            rows = _table_rows(page, fitz.Rect(b[:4]))
            if rows:
                out[int(m.group(1))] = rows
    return out


def tabular(rows):
    ncol = max(len(r) for r in rows)
    body = " \\\\\n".join(" & ".join(esc(c) for c in r + [""] * (ncol - len(r))) for r in rows)
    return ("\\begin{tabular}{" + "l" * ncol + "}\n\\toprule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}")


def clean(t, vocab=None):
    t = "\n".join(l for l in t.split("\n") if not LINENO.match(l))
    return dehyphenate(t, vocab) if vocab else t


def esc(s):
    # an unescaped % would comment out the rest of the line for every reader downstream
    return s.replace("\\", " ").replace("{", "(").replace("}", ")").replace("$", " ").replace("%", "\\%")


def mark_pointers(s):
    def rep(m):
        k = KIND.get(m.group(1).lower())
        return f"{m.group(1)} \\ref{{{k}:{m.group(2)}}}" if k else m.group(0)
    return POINTER.sub(rep, s)


def _key(t):
    return re.sub(r"[^A-Za-z0-9]+", "", t)[:40] or "anon"


def mark_cites(s):
    def citet(m):
        yrs = [y.strip() for y in m.group(2).split(",")]
        return m.group(1) + " \\cite{" + ",".join(_key(m.group(1) + y) for y in yrs) + "}"
    s = CITET.sub(citet, s)

    def brak(m):
        inner = m.group(1)
        parts = re.split(r",\s*(?=[A-Z])", inner)
        keys = [_key(p) for p in parts if re.search(r"(19|20)\d\d", p)] or [_key(inner)]
        return "\\cite{" + ",".join(keys) + "}"
    s = BRACKET_AY.sub(brak, s)

    def num(m):
        keys = [x.strip() for x in re.split(r"[,\-–]", m.group(1)) if x.strip().isdigit()]
        return "\\cite{" + ",".join(f"ref{k}" for k in keys) + "}" if keys else m.group(0)
    s = NUMCITE.sub(num, s)

    def paren(m):
        return "\\cite{" + _key(m.group(1)) + "}"
    return PAREN_AY.sub(paren, s)


def convert(meta_path):
    m = json.load(open(meta_path))
    secs = m.get("sections") or {}
    code = "A4S" + str(m["submission_number"]).zfill(4)
    pdf = os.path.join(os.path.dirname(meta_path), "paper.pdf")
    printed = sections_from_pdf(pdf)
    printed = [(lv, t, "\n".join(ls)) for lv, t, ls in printed if not FRONT.match(t)]
    if len(printed) < 4 and not secs:
        return None
    tabs = pdf_tables(pdf)
    order = [k for k in secs if k not in BODY_SKIP]
    vocab = vocabulary(secs)
    parts = ["% Generated from the submission PDF by benchA4S/scripts/pdf_to_tex.py.",
             "% Structure, floats, citations and cross-references are recovered from the rendered",
             "% text; anything the authors did not write in the PDF is not present here.",
             "\\documentclass{article}", "\\begin{document}", "\\maketitle"]
    if secs.get("abstract"):
        parts += ["\\begin{abstract}", esc(clean(secs["abstract"], vocab)), "\\end{abstract}"]
    # the printed headings when they were found, the canonical buckets only as a fallback
    units = ([(lv, t, b) for lv, t, b in printed
              if not any(w in t.lower() for w in BODY_SKIP_WORDS)] if len(printed) >= 4
             else [(1, k.replace("_", " ").title(), secs[k]) for k in order])
    floats, nsec = set(), 0
    for lv, title, raw in units:
        body = clean(raw, vocab)
        nsec += 1
        cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[lv]
        chunk = []
        pos = 0
        for c in CAPTION.finditer(body):
            kind = KIND.get(c.group(1).lower(), "fig")
            num = c.group(2)
            env = {"fig": "figure", "tab": "table", "alg": "algorithm"}.get(kind, "figure")
            tail = body[c.end():c.end() + 240].split("\n\n")[0]
            chunk.append(esc(body[pos:c.start()]))
            rows = tabs.get(int(num)) if kind == "tab" else None
            inner = tabular(rows) + "\n" if rows else ""
            chunk.append(f"\\begin{{{env}}}\n{inner}\\caption{{{esc(tail)}}}\n\\label{{{kind}:{num}}}\n\\end{{{env}}}")
            floats.add(f"{kind}:{num}")
            pos = c.end() + len(tail)
        chunk.append(esc(body[pos:]))
        text = "\n".join(chunk)
        text = mark_pointers(text)
        text = mark_cites(text)
        parts += [f"\\{cmd}{{{esc(title)}}}", f"\\label{{sec:{nsec}}}", text]
    parts += ["\\bibliography{refs}", "\\end{document}"]
    d = os.path.join(OUT, code)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "main.tex"), "w").write("\n\n".join(parts))
    return code


def main():
    import glob
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(f"{A4S}/papers/*/*/meta.json")):
        if convert(p):
            n += 1
    print(f"PDF -> tex converted {n} papers -> {OUT}")


if __name__ == "__main__":
    main()
