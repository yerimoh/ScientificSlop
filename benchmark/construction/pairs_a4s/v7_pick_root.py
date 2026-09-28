"""Choose each A4S paper's tex root by how much of its own PDF the source actually contains.

Title match alone is not enough. A4S0218's zip holds several generations of the manuscript and
A4S0272's holds a short draft that shares the title and the abstract but not the body. The rule
here is the one that matters for a benchmark: the source must be the document that was submitted,
so the root is the complete tex whose text covers the most sentences of the submitted PDF.

Output: cache/a4s_tex_index.json (root, title_sim, pdf_sentence_coverage, metrics)
"""
import difflib, glob, json, os, re, sys

import fitz

ROOT = os.environ.get("SCISLOP_ROOT", ".")
A = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"
sys.path.insert(0, f"{ROOT}/paper/draft_v6/scislopbench/data/scripts")
from slopbench_lib import flatten, metrics, strip_comments  # noqa: E402

CAP = re.compile(r"^(figure|table|algorithm)\s*\d", re.I)


def norm(s):
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def tex_title(s):
    for m in re.finditer(r"\\title\s*(?:\[[^\]]*\])?\s*\{", s):
        i, depth, buf = m.end(), 1, []
        while i < len(s) and depth:
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if not depth:
                    break
            buf.append(ch)
            i += 1
        t = re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+|[{}\\]", " ", "".join(buf))).strip()
        if len(t.split()) >= 3:
            return t
    return ""


def pdf_sentences(pdf, n=40):
    t = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", "\n".join(p.get_text() for p in fitz.open(pdf)))
    cut = t.lower().rfind("\nreferences")
    if cut > 0:
        t = t[:cut]
    return [s for s in re.split(r"(?<=[.!?])\s+", t)
            if len(s.split()) >= 12 and not CAP.match(s.strip())][:n]


def coverage(tex, sents):
    T = norm(re.sub(r"\\[a-zA-Z@]+\*?|[{}$\\]", " ", tex))
    hit = 0
    for s in sents:
        w = norm(s).split()
        if any(" ".join(w[i:i + 6]) in T for i in range(0, max(1, len(w) - 5))):
            hit += 1
    return hit / len(sents) if sents else 0.0


def main():
    idx = json.load(open(f"{A}/cache/a4s_index.json"))
    out = {}
    for d in sorted(glob.glob(f"{A}/a4s_tex/A4S*") + glob.glob(f"{A}/a4s_arxiv/A4S*")):
        code = os.path.basename(d).split("_")[0]
        if code not in idx:
            print(f"  {code}: not in index, skipping")
            continue
        sents = pdf_sentences(os.path.join(ROOT, idx[code]["dir"], "paper.pdf"))
        best = None
        for f in glob.glob(os.path.join(d, "**", "*.tex"), recursive=True):
            raw = open(f, encoding="utf-8", errors="ignore").read()
            if not (re.search(r"\\documentclass", raw) and re.search(r"\\begin\{document\}", raw)):
                continue
            try:
                full = flatten(f)
                m = metrics(full)
            except Exception:
                continue
            sim = difflib.SequenceMatcher(None, norm(tex_title(strip_comments(full))),
                                          norm(idx[code]["title"])).ratio()
            if sim < 0.9:
                continue
            cov = coverage(full, sents)
            key = (round(cov, 3), m["body_words"])
            if best is None or key > best[0]:
                best = (key, f, sim, cov, m)
        if best:
            _, f, sim, cov, m = best
            out[code] = dict(root=os.path.relpath(f, A), title_sim=round(sim, 3),
                             pdf_sentence_coverage=round(cov, 3), n_pdf_sentences=len(sents),
                             metrics=m)
    json.dump(out, open(f"{A}/cache/a4s_tex_index.json", "w"), indent=1)
    print(f"{'code':9}{'title':>6}{'PDFmatch':>8}{'body':>7}  root")
    for c, v in sorted(out.items()):
        mark = "" if v["pdf_sentence_coverage"] >= 0.8 else "  <- low"
        print(f"{c:9}{v['title_sim']:>6.2f}{v['pdf_sentence_coverage']:>8.2f}{v['metrics']['body_words']:>7}  "
              f"{v['root'][:54]}{mark}")


if __name__ == "__main__":
    main()
