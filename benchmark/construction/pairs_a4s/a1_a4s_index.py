"""Step 1 (Agents4Science): index every A4S 2025 AI/ML paper.

The A4S side has no TeX, only the submission PDF, so every structural metric that the FARS
pipeline reads out of the flattened tex is re-derived from the PDF here:
  body_words        prose words of the body sections (abstract/references/appendix/disclosure out)
  n_sections        detected top-level headings of the body
  n_figures/tables  distinct "Figure N"/"Table N" caption numbers
  n_unique_cites    distinct citation markers used in the body ([12] or (Author, 2024))
  n_internal_refs   in-text pointers to a numbered float or section
References are parsed from the PDF reference section for all 154 papers (references_verified
in meta.json exists for only 61, so it is used as a title/arXiv-id hint, not as the pool).
cite_roles maps a reference index to the normalized sections that cite it.

Output: cache/a4s_index.json
"""
import os, re, sys, json, glob, zipfile

import fitz  # PyMuPDF

ROOT = os.environ.get("SCISLOP_ROOT", ".")
A4S = f"{ROOT}/Agents4Science/2025_full"
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"

# meta.json section keys -> the five roles the pair rule uses
SEC_ROLE = {
    "introduction": "introduction", "abstract": "other",
    "background": "related_work", "related": "related_work",
    "method": "method",
    "experiments": "experiments", "results": "experiments", "ablation": "experiments",
    "analysis": "experiments", "evaluation": "experiments",
    "discussion": "conclusion", "conclusion": "conclusion", "limitations": "conclusion",
    "future_work": "conclusion",
}
BODY_SKIP = {"abstract", "references", "appendix", "ai_disclosure", "acknowledgments",
             "acknowledgements", "checklist", "broader_impacts", "ethics"}
# everything from the first page down to the reference list, which is what split_body() keeps on
# the TeX side. Reading the PDF straight through instead of stitching the per-section texts back
# together matters: the section split covers only ~93% of the text (min 59%), and floats that sit
# after the reference heading were being dropped.
BODY_END_SEC = {"references", "appendix", "checklist", "acknowledgments", "acknowledgements"}
COUNTED_SEC = {"abstract", "references"}          # not \section{} on the TeX side either

LINENO = re.compile(r"^\s*\d{1,4}\s*$")          # submission line numbers sit on their own line
NUMREF = re.compile(r"^\[(\d{1,3})\]\s*")
ARXIV = re.compile(r"arxiv[:\s/]*(\d{4}\.\d{4,5})", re.I)
CHECKLIST = re.compile(r"(?:Agents4Science\s+AI\s+Involvement\s+Checklist|IMPORTANT, please|NeurIPS Paper Checklist|^\s*A\s+Appendix)", re.M | re.I)
ENTRY_END = re.compile(r"(?:19|20)\d{2}[a-z]?\s*[.)]\s*(?:doi:\s*\S+\s*|URL\s+\S+\s*|Available[^.]{0,40}\.\s*)*(?=[A-ZÀ-Þ])")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def clean(t):
    """Drop the standalone line numbers the A4S template prints in the margin."""
    return "\n".join(l for l in t.split("\n") if not LINENO.match(l))


def prose_words(t):
    return len(WORD.findall(clean(t)))


def pdf_body(pdf_path, headings):
    """(body text, appendix text, number of body headings) read straight off the PDF.

    headings are meta.json's headings_detected, whose line indices address the same fitz
    extraction; the body ends at the first reference/appendix heading.
    """
    lines = "\n".join(pg.get_text() for pg in fitz.open(pdf_path)).split("\n")
    hs = [h for h in headings if 0 <= h["line"] < len(lines)]
    ends = [h["line"] for h in hs if h.get("section_name") in BODY_END_SEC]
    end = min(ends) if ends else len(lines)
    n_sec = len({h["line"] for h in hs if h["line"] < end and h.get("section_name") not in COUNTED_SEC})
    # the appendix is what follows an appendix heading. The rendered reference list is thousands
    # of words in a PDF and nothing on the TeX side, so it is not counted on either.
    app = [h["line"] for h in hs if h.get("section_name") == "appendix" and h["line"] >= end]
    tail = "\n".join(lines[min(app):]) if app else ""
    return "\n".join(lines[:end]), tail, n_sec


def parse_refs(reftext):
    """Reference section -> [{idx, raw, title, arxiv}].

    Two entry layouts appear in the corpus. Numbered lists split on a strictly increasing [N].
    Unnumbered (plainnat/apa) lists have no reliable indentation left in the PDF text, so entries
    are split after a year that terminates an entry and is followed by a fresh capitalized name.
    """
    t = clean(reftext)
    t = CHECKLIST.split(t)[0]
    pos = [(m.start(), int(m.group(1))) for m in re.finditer(r"\[(\d{1,3})\]", t)]
    pos = [(p, n) for p, n in pos if n <= 300]
    seq, want = [], 1
    for p, n in pos:
        if n == want:
            seq.append((p, n))
            want += 1
    entries = []
    if len(seq) >= 3:
        for i, (p, n) in enumerate(seq):
            end = seq[i + 1][0] if i + 1 < len(seq) else len(t)
            entries.append((n, t[p:end]))
    else:
        flat = re.sub(r"\s+", " ", t)
        cuts = [0] + [m.end() for m in ENTRY_END.finditer(flat)] + [len(flat)]
        chunks = [flat[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
        entries = [(i + 1, c) for i, c in enumerate(chunks)]
    refs = []
    for n, raw in entries:
        raw = re.sub(r"\s+", " ", re.sub(r"^\[\d{1,3}\]\s*", "", raw)).strip()
        raw = re.sub(r"-\s+(?=[a-z])", "", raw)          # undo PDF hyphenation
        # a URL broken across a space by the PDF extractor spills into the next entry; drop it
        raw = re.sub(r"^(?:URL\s+|doi:\s*|Available at\s+|https?://)\S*(?:\s+[a-z0-9._\-/~%?=]+)*[.,]?\s*",
                     "", raw)
        if len(raw.split()) < 5:
            continue
        am = ARXIV.search(raw)
        refs.append(dict(idx=n, raw=raw[:600], title=guess_title(raw), arxiv=am.group(1) if am else None))
    return refs


TITLE_STOP = re.compile(                              # case-sensitive on purpose: lowercase
    r"\b(In\s|Proceedings|Advances in|Conference|Workshop|Journal|Transactions|"   # "in" inside a
    r"[Aa]rXiv|IEEE|ACM|preprint|URL\s|https?://|pp\.|vol\.|Vol\.|volume\s)")   # title is normal

LOWER_OK = {"and", "van", "von", "der", "de", "di", "del", "la", "le", "el", "et", "al", "dos", "da", "bin"}
ABBREV = r"(?<!\b[A-Z])(?<!\bInc)(?<!\bLtd)(?<!\bJr)(?<!\bpp)(?<!\bvol)(?<!\beds)(?<!\bNo)"


def _sentences(s):
    parts = re.split(ABBREV + r"[.?]\s+", s)
    return [x.strip() for x in parts if x.strip()]


def _author_like(sent):
    w = re.findall(r"[A-Za-zÀ-ÿ'\-]+", sent)
    if not w:
        return True
    low = sum(1 for x in w if x[0].islower() and x.lower() not in LOWER_OK)
    return low / len(w) < 0.25


def guess_title(raw):
    """First sentence of the entry that reads like a title rather than an author list or a venue."""
    s = raw
    m = re.match(r"^(.*?\(\d{4}[a-z]?\)\.?)\s*(.+)$", s)          # APA: Authors (2015). Title. Venue
    if m:
        s = m.group(2)
    for sent in _sentences(s):
        if _author_like(sent):
            continue
        st = TITLE_STOP.search(sent)
        if st is not None and st.start() < 15:
            continue
        if st is not None:
            sent = sent[:st.start()]
        sent = re.sub(r"\s+", " ", sent).strip(" .,;:")
        sent = re.sub(r"^(In|Technical [Rr]eport|Preprint)\s+", "", sent)
        sent = re.sub(r",?\s*(?:19|20)\d\d[a-z]?$", "", sent).strip(" .,;:")
        if 3 <= len(sent.split()) <= 30:
            return sent
    return ""


def first_author_year(raw):
    """(surname, year) of a reference entry, for author-year citation matching."""
    ym = re.findall(r"\b(19|20)(\d\d)[a-z]?\b", raw)
    year = (ym[0][0] + ym[0][1]) if ym else None
    m = re.match(r"^([A-ZÀ-Þ][A-Za-zÀ-ÿ'\-]{1,})\s*,", raw)                  # Surname, A.
    if not m:
        m = re.match(r"^(?:[A-ZÀ-Þ][A-Za-zÀ-ÿ'\-]*\.?\s+){1,3}([A-ZÀ-Þ][A-Za-zÀ-ÿ'\-]{2,})\b", raw)
    return (m.group(1) if m else None), year


def cite_markers(text):
    """Return (set of numeric refs cited, count of citation commands) for one body section."""
    t = clean(text)
    nums, ncmd = set(), 0
    for m in re.finditer(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]", t):
        ncmd += 1
        for part in re.split(r"[,–\-]", m.group(1)):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= 300:
                nums.add(int(part))
    ay = re.findall(r"\(([^()]{0,80}?\b(?:19|20)\d\d[a-z]?)\)", t)
    return nums, ncmd, len(ay)


def main():
    idx = {}
    for mp in sorted(glob.glob(f"{A4S}/papers/*/*/meta.json")):
        m = json.load(open(mp))
        code = "A4S" + str(m["submission_number"]).zfill(4)
        secs = m.get("sections") or {}
        body_keys = [k for k in secs if k not in BODY_SKIP]
        body, tail, n_sec = pdf_body(os.path.join(os.path.dirname(mp), "paper.pdf"),
                                     m.get("headings_detected") or [])
        if not body.strip():                       # no usable headings, fall back to the sections
            body = "\n".join(secs[k] for k in body_keys)
            tail = secs.get("appendix", "")
            n_sec = len(body_keys)
        refs = parse_refs(secs.get("references", ""))
        # section -> cited reference numbers
        roles, nums_all, ncmd, nay = {}, set(), 0, 0
        for k in body_keys:
            nums, nc, na = cite_markers(secs[k])
            ncmd += nc
            nay += na
            nums_all |= nums
            role = SEC_ROLE.get(k, "other")
            for n in nums:
                roles.setdefault(str(n), {}).setdefault(role, 0)
                roles[str(n)][role] += 1
        # author-year papers carry no [N]; match "Surname ... (year)" back to the reference
        for r in refs:
            sur, yr = first_author_year(r["raw"])
            r["first_author"], r["year"] = sur, yr
            if not (sur and yr) or str(r["idx"]) in roles:
                continue
            pat = re.compile(r"\b" + re.escape(sur) + r"\b[^.]{0,80}?\b" + yr + r"\b")
            for k in body_keys:
                n = len(pat.findall(clean(secs[k])))
                if n:
                    role = SEC_ROLE.get(k, "other")
                    roles.setdefault(str(r["idx"]), {}).setdefault(role, 0)
                    roles[str(r["idx"])][role] += n
        figs = {int(x) for x in re.findall(r"Figure\s+(\d{1,2})\s*[:.]", clean(body))}
        tabs = {int(x) for x in re.findall(r"Table\s+(\d{1,2})\s*[:.]", clean(body))}
        internal = len(re.findall(r"\b(?:Figure|Fig\.|Table|Section|Sec\.|Equation|Eq\.|Algorithm|Appendix)\s*~?\s*"
                                  r"(?:\d{1,2}|[A-D])\b", clean(body)))
        # 22 of the 154 submissions ship LaTeX inside supplementary.zip; the slop checkers read
        # TeX on both sides, so record which A4S papers can be measured symmetrically at all.
        supp = os.path.join(os.path.dirname(mp), "supplementary.zip")
        tex_files = []
        if os.path.isfile(supp):
            try:
                tex_files = [n for n in zipfile.ZipFile(supp).namelist() if n.lower().endswith(".tex")]
            except Exception:
                tex_files = []
        rv = m.get("references_verified") or []
        idx[code] = dict(
            code=code, submission_id=m["submission_id"], submission_number=m["submission_number"],
            status=m["status"], dir=os.path.relpath(os.path.dirname(mp), ROOT),
            title=m["title"], abstract=(m.get("abstract") or "")[:4000],
            keywords=m.get("keywords") or [],
            scores=m.get("scores"), autonomy=m.get("autonomy"),
            metrics=dict(body_words=prose_words(body), n_sections=n_sec,
                         section_keys=body_keys, n_figures=len(figs), n_tables=len(tabs),
                         n_refs=len(refs), n_unique_cites=len(nums_all) or None,
                         n_cite_cmds=ncmd, n_authoryear=nay, n_internal_refs=internal,
                         appendix_words=prose_words(tail)),
            tex_in_supplementary=bool(tex_files), n_tex_files=len(tex_files),
            refs=refs, cite_roles=roles,
            verified_hint={str(i + 1): dict(
                title=(r.get("matched") or {}).get("title") or (r.get("parsed") or {}).get("title"),
                arxiv=((r.get("matched") or {}).get("external_ids") or {}).get("ArXiv")
                      or (r.get("parsed") or {}).get("arxiv_id"),
                status=r.get("status")) for i, r in enumerate(rv)} if rv else {},
        )
    json.dump(idx, open(f"{OUTD}/cache/a4s_index.json", "w"), ensure_ascii=False)
    import statistics as st
    bw = [r["metrics"]["body_words"] for r in idx.values()]
    nr = [len(r["refs"]) for r in idx.values()]
    nc = [r["metrics"]["n_unique_cites"] or 0 for r in idx.values()]
    print(f"papers {len(idx)} | body_words med {st.median(bw):.0f} min {min(bw)} max {max(bw)}")
    print(f"refs parsed total {sum(nr)} med {st.median(nr)} | zero-ref papers {sum(1 for n in nr if n == 0)}")
    print(f"in-text numeric cites med {st.median(nc)} | papers with none {sum(1 for n in nc if n == 0)}")


if __name__ == "__main__":
    main()
