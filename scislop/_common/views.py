"""Document views built from LaTeX, in document order, with offset maps.

This module replaces `temp_common.ai_tex()` (main.tex + sorted(glob(sections/*.tex))),
which concatenated section files in alphabetical order and therefore destroyed the
document order that every structural measurement depends on. Here \\input/\\include are
expanded at the position where they occur.

Lessons folded in (from artifact-ai2science failure notes):
  * comments are stripped with (?<!\\\\)% so that "5.39\\%" keeps its number;
  * float environments are removed from prose but their tables are parsed separately;
  * math, citations and references become alphabetic sentinels (xxmathxx, xxcitexx,
    xxrefxx) that SURVIVE [A-Za-z]+ tokenisation. The old "[CITE]" sentinel was stripped
    by the tokenizer and silently disabled the citation filter;
  * reference-like macros are matched broadly (\\[A-Za-z]*[Rr]ef[A-Za-z]*) so template
    macros such as \\Secref are not lost; names like \\href, \\crefname, \\refstepcounter
    are blacklisted;
  * every derived string carries an `origin` array so that any span can be mapped back to
    the expanded tex offset. Gold loci are always expressed in expanded-tex offsets.
"""
from __future__ import annotations
import csv
import os
import re
from dataclasses import dataclass, field

from corpus import read

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- mapped text

class MappedText:
    """A string plus, for each character, the offset it came from in the source tex."""

    def __init__(self, text: str, origin: list[int] | None = None):
        self.text = text
        self.origin = origin if origin is not None else list(range(len(text)))
        assert len(self.text) == len(self.origin)

    def sub(self, pattern, repl, flags=0) -> "MappedText":
        """Regex substitution; replacement characters inherit the match start offset.
        `repl` may be a plain string (no back-references) or a callable(match) -> str."""
        rx = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
        out, org, last = [], [], 0
        for m in rx.finditer(self.text):
            out.append(self.text[last:m.start()]); org.extend(self.origin[last:m.start()])
            r = repl(m) if callable(repl) else repl
            anchor = self.origin[m.start()] if m.start() < len(self.origin) else (self.origin[-1] if self.origin else 0)
            out.append(r); org.extend([anchor] * len(r))
            last = m.end()
        out.append(self.text[last:]); org.extend(self.origin[last:])
        return MappedText("".join(out), org)

    def slice(self, a: int, b: int) -> "MappedText":
        return MappedText(self.text[a:b], self.origin[a:b])

    def tex_span(self, a: int, b: int) -> tuple[int, int]:
        """Map a [a, b) span of this view to a [start, end) span of the source tex."""
        if not self.origin:
            return (0, 0)
        a = max(0, min(a, len(self.origin) - 1))
        b = max(a + 1, min(b, len(self.origin)))
        return (self.origin[a], self.origin[b - 1] + 1)


# --------------------------------------------------------------------------- expansion

def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
MACRO_FILE_RE = re.compile(r"math_commands|macros|preamble|commands|packages|defs", re.I)


def _resolve(base: str, name: str) -> str | None:
    name = name.strip()
    cands = [name, name + ".tex", name.replace("/", "__"), name.replace("/", "__") + ".tex",
             os.path.basename(name), os.path.basename(name) + ".tex"]
    for c in cands:
        p = os.path.join(base, c)
        if os.path.isfile(p):
            return p
    return None


def expand_tex(main_path: str, depth: int = 4):
    """Return (expanded_tex, report). Comments are stripped file by file before expansion so
    a commented-out \\input is not expanded. Macro files are expanded too (we need their
    \\def\\Secref-style definitions in scope), but they contain no sections or prose."""
    base = os.path.dirname(main_path)
    report = {"expanded": [], "missing": [], "macro_files": []}
    seen = set()

    def go(path: str, d: int) -> str:
        if d > depth or path in seen:
            return ""
        seen.add(path)
        txt = strip_comments(read(path))

        def rep(m):
            target = _resolve(base, m.group(1))
            if not target:
                report["missing"].append(m.group(1))
                return f"\n% MISSING INPUT {m.group(1)}\n"
            if MACRO_FILE_RE.search(m.group(1)):
                report["macro_files"].append(target)
            report["expanded"].append(target)
            return "\n" + go(target, d + 1) + "\n"
        return INPUT_RE.sub(rep, txt)

    return go(main_path, 0), report


# --------------------------------------------------------------------------- sections

def _load_role_map():
    rows = list(csv.DictReader(open(os.path.join(HERE, "section_role_map.tsv")), delimiter="\t"))
    return [(r["role"], re.compile(r["pattern"], re.I)) for r in rows]


ROLE_MAP = _load_role_map()


def clean_title(t: str) -> str:
    t = re.sub(r"\\label\s*\{[^{}]*\}", " ", t)          # a label inside the title must not leak into the role match
    t = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$~]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def role_of(title: str) -> str:
    t = clean_title(title).lower()
    for role, rx in ROLE_MAP:
        if rx.search(t):
            return role
    return "Other"


SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
FLOAT_ENV_RE = re.compile(
    r"\\begin\{(figure\*?|table\*?|algorithm\*?|algorithmic|wrapfigure|wraptable|sidewaystable\*?|"
    r"tabular\*?|tabularx|longtable|minipage|tcolorbox|lstlisting|verbatim|Verbatim|minted|framed|mdframed|shaded|"
    r"promptbox|prompt|tikzpicture|listing|boxedminipage|comment)\}(.*?)\\end\{\1\}", re.S)
MATH_ENV_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|"
                         r"flalign\*?|alignat\*?)\}(.*?)\\end\{\1\}", re.S)
# reference-like macro names, broad on purpose (see module docstring), with a blacklist
REF_MACRO_RE = re.compile(r"\\([A-Za-z]*[Rr]ef[A-Za-z]*)\s*((?:\{[^{}]*\}\s*)+)")
REF_BLACKLIST = re.compile(r"href|name$|format|stepcounter|Fraction|hyperref|refstyle|prefix|url|newref|labelref|reftitle", re.I)
CITE_RE = re.compile(r"\\[cC]ite[A-Za-z*]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")
APPENDIX_RE = re.compile(r"\\appendix\b|\\section\*?\{\s*(?:Appendix|Appendices|Supplementary|Acknowledg)", re.I)
# fallback for papers that start their appendix without the \appendix macro: the first top-level
# section after the last Conclusion-role section whose title is appendix-like
APPENDIX_TITLE_RE = re.compile(r"appendix|supplement|acknowledg|implementation details?|additional (?:results?|experiments?|details?|analys)|"
                               r"further (?:results?|details?|analys)|proofs?|derivations?|extended|broader impact|checklist|ethic", re.I)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)


@dataclass
class Section:
    idx: int            # top-level section index (0-based, document order)
    level: int          # 1 section, 2 subsection, 3 subsubsection
    title: str
    role: str
    start: int          # offset of the heading in expanded tex
    body_start: int
    end: int            # exclusive
    parent: int         # top-level index this heading belongs to
    labels: list = field(default_factory=list)
    in_appendix: bool = False


@dataclass
class Doc:
    corpus: str
    id: str
    tex: str                       # expanded, comments stripped
    doc_start: int                 # \begin{document} offset (0 if absent)
    doc_end: int
    appendix_start: int | None
    sections: list                 # all headings (levels 1-3)
    top_sections: list             # level-1 only (document-order), including appendix ones
    labels: dict                   # label -> {"kind", "section": top idx, "offset"}
    abstract: str
    report: dict
    n_words_body: int

    def body_sections(self):
        """Top-level sections before the appendix, excluding acknowledgements."""
        return [s for s in self.top_sections if not s.in_appendix and s.role != "Appendix"]

    def body_end(self) -> int:
        return self.appendix_start if self.appendix_start else self.doc_end

    def top_of(self, offset: int):
        for s in self.top_sections:
            if s.start <= offset < s.end:
                return s.idx
        return None


def _label_kind(tex: str, pos: int, floats, maths) -> str:
    for a, b, env in floats:
        if a <= pos < b:
            if env.startswith("table") or "tabular" in env or "longtable" in env:
                return "table"
            if env.startswith("algorithm"):
                return "algorithm"
            return "figure"
    for a, b in maths:
        if a <= pos < b:
            return "equation"
    win = tex[max(0, pos - 200):pos]
    if re.search(r"\\begin\{(theorem|lemma|proposition|corollary|definition|assumption|remark)", win):
        return "theorem"
    if re.search(r"\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\{[^{}]*\}\s*$", win.rstrip()) or re.search(r"\\(?:sub)*section", win[-120:]):
        return "section"
    return "section"


def load_doc(paper: dict) -> Doc:
    tex, report = expand_tex(paper["main_tex"])
    m0 = re.search(r"\\begin\{document\}", tex)
    m1 = re.search(r"\\end\{document\}", tex)
    doc_start = m0.end() if m0 else 0
    doc_end = m1.start() if m1 else len(tex)

    heads = [m for m in SECTION_RE.finditer(tex) if doc_start <= m.start() < doc_end]
    app = APPENDIX_RE.search(tex, doc_start, doc_end)
    appendix_start = app.start() if app else None

    sections = []
    top_idx = -1
    for i, m in enumerate(heads):
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)]
        end = heads[i + 1].start() if i + 1 < len(heads) else doc_end
        if level == 1:
            top_idx += 1
        title = clean_title(m.group(2))
        sec = Section(idx=top_idx if level == 1 else len(sections), level=level, title=title,
                      role=role_of(title), start=m.start(), body_start=m.end(), end=end,
                      parent=max(top_idx, 0),
                      in_appendix=bool(appendix_start is not None and m.start() >= appendix_start))
        sections.append(sec)
    top = [s for s in sections if s.level == 1]
    for j, s in enumerate(top):
        s.end = top[j + 1].start if j + 1 < len(top) else doc_end

    if appendix_start is None and top:
        concl = [j for j, s_ in enumerate(top) if s_.role == "Conclusion"]
        after = top[concl[-1] + 1:] if concl else []
        cand = next((s_ for s_ in after if APPENDIX_TITLE_RE.search(s_.title)), None)
        if cand is not None:
            appendix_start = cand.start
            for s_ in top:
                s_.in_appendix = s_.start >= appendix_start
            for s_ in sections:
                s_.in_appendix = s_.start >= appendix_start

    floats = [(m.start(), m.end(), m.group(1)) for m in FLOAT_ENV_RE.finditer(tex)]
    maths = [(m.start(), m.end()) for m in MATH_ENV_RE.finditer(tex)]
    labels = {}
    for m in LABEL_RE.finditer(tex):
        if not (doc_start <= m.start() < doc_end):
            continue
        kind = _label_kind(tex, m.start(), floats, maths)
        key = m.group(1).strip()
        if key in labels:
            continue
        labels[key] = {"kind": kind, "section": None, "offset": m.start()}
        for s in top:
            if s.start <= m.start() < s.end:
                labels[key]["section"] = s.idx
                if kind == "section":
                    s.labels.append(key)
                break
    am = ABSTRACT_RE.search(tex)
    abstract = am.group(1) if am else ""
    body_end = appendix_start if appendix_start else doc_end
    n_words = len(re.findall(r"[A-Za-z]+", prose_view(tex[doc_start:body_end]).text))
    return Doc(corpus=paper["corpus"], id=paper["id"], tex=tex, doc_start=doc_start, doc_end=doc_end,
               appendix_start=appendix_start, sections=sections, top_sections=top, labels=labels,
               abstract=abstract, report=report, n_words_body=n_words)


# --------------------------------------------------------------------------- prose view

SENT_MATH, SENT_CITE, SENT_REF = "xxmathxx", "xxcitexx", "xxrefxx"
SENTINELS = {SENT_MATH, SENT_CITE, SENT_REF, "xxcaptionxx", "xxverbatimxx"}
FORMAT_MACROS = r"emph|textbf|textit|texttt|textsc|underline|text|mathrm|textrm|textnormal|small|footnotesize|url|texorpdfstring|mbox"
VERBATIM_ENV_RE = re.compile(
    r"\\begin\{(tcolorbox|tcblisting|lstlisting|verbatim|Verbatim|minted|mdframed|framed|alltt|promptbox|listing|"
    r"boxedverbatim|shaded|tabbing|exampleblock|prompt|chat|dialogue|quote|quotation|verse|lstinputlisting)\*?\}(?:\[[^\]]*\])?(.*?)\\end\{\1\*?\}", re.S)
CAP_IN_FLOAT_RE = re.compile(r"\\caption\*?\s*(?:\[[^\]]*\])?\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.S)


def prose_view(tex: str, keep_captions: bool = False, base: int = 0) -> MappedText:
    """Turn (a slice of) expanded tex into running prose with an offset map.

    Floats are removed (captions optionally kept as `xxcaptionxx <text>`), math, cites and refs
    become sentinels, remaining macros keep only their argument text, braces vanish.
    """
    mt = MappedText(tex, [base + i for i in range(len(tex))])
    # the loader's own marker for an unresolved \input is inserted after comment stripping; it is not prose
    mt = mt.sub(r"% MISSING INPUT[^\n]*", " ")
    mt = mt.sub(VERBATIM_ENV_RE, " xxverbatimxx ")
    if keep_captions:
        mt = mt.sub(FLOAT_ENV_RE, lambda m: " ".join(
            " xxcaptionxx " + c for c in CAP_IN_FLOAT_RE.findall(m.group(2))) + " ")
    else:
        mt = mt.sub(FLOAT_ENV_RE, " ")
    mt = mt.sub(MATH_ENV_RE, f" {SENT_MATH} ")
    mt = mt.sub(r"\$\$.*?\$\$", f" {SENT_MATH} ", re.S)
    mt = mt.sub(r"\\\[.*?\\\]", f" {SENT_MATH} ", re.S)
    mt = mt.sub(r"\$[^$]*\$", f" {SENT_MATH} ")
    mt = mt.sub(CITE_RE, f" {SENT_CITE} ")
    mt = mt.sub(REF_MACRO_RE, lambda m: m.group(0) if REF_BLACKLIST.search(m.group(1)) else f" {SENT_REF} ")
    mt = mt.sub(r"\\(?:SetKw[A-Za-z]*|SetAlgo[A-Za-z]*|DontPrintSemicolon|newcommand|renewcommand|providecommand|DeclareMathOperator|newtheorem|newenvironment|"
                r"definecolor|setcounter|addtocounter|newcounter|pagestyle|thispagestyle)\*?\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}\s*){1,4}(?:\[[^\]]*\])?", " ")
    mt = mt.sub(r"\\(?:label|includegraphics|vspace|hspace|footnote|footnotemark|caption|newline|centering|noindent|"
                r"graphicspath|bibliography|bibliographystyle|maketitle|setlength|linewidth|input|include)\*?\s*(?:\[[^\]]*\])?\s*(?:\{(?:[^{}]|\{[^{}]*\})*\})?", " ")
    mt = mt.sub(r"\\(?:sub)*section\*?\s*(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", lambda m: "\n\n" + m.group(1) + ".\n\n")
    mt = mt.sub(r"\\paragraph\*?\{((?:[^{}]|\{[^{}]*\})*)\}", lambda m: "\n\n" + m.group(1) + ". ")
    mt = mt.sub(r"\\begin\{(?:itemize|enumerate|description|abstract|quote|center|compactitem|compactenum)\}(?:\s*\[[^\]]*\])?|\\end\{(?:itemize|enumerate|description|abstract|quote|center|compactitem|compactenum)\}", "\n")
    mt = mt.sub(r"\\item\b", "\n")
    mt = mt.sub(r"\\begin\{[A-Za-z*]+\}\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})?|\\end\{[A-Za-z*]+\}", "\n")
    mt = mt.sub(rf"\\(?:{FORMAT_MACROS})\*?\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}", lambda m: m.group(1))
    mt = mt.sub(r"\\[A-Za-z]+\*?\s*(?:\[[^\]]*\])?", " ")
    mt = mt.sub(r"[{}~]", " ")
    mt = mt.sub(r"\\%", "%")
    mt = mt.sub(r"\\&", "&")
    mt = mt.sub(r"[ \t]+", " ")
    mt = mt.sub(r"\n{3,}", "\n\n")
    return mt


# --------------------------------------------------------------------------- sentences

ABBR_RE = re.compile(r"\b(et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Tabs|resp|approx|no|etc|al|Dr|Prof|w\.r\.t|a\.k\.a|Alg|App)\.$", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'“])")


def sentences(mt: MappedText) -> list[dict]:
    """Split a prose MappedText into sentences with view offsets and tex spans.
    Paragraph breaks always split. Abbreviations are protected."""
    out = []
    for pm in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", mt.text):
        para = pm.group(0)
        base = pm.start()
        pieces, pos = [], 0
        for sm in SENT_SPLIT_RE.finditer(para):
            seg = para[pos:sm.start()]
            if pieces and ABBR_RE.search(pieces[-1][2]):
                pieces[-1] = (pieces[-1][0], sm.start(), pieces[-1][2] + para[pieces[-1][1]:sm.start()])
            else:
                pieces.append((pos, sm.start(), seg))
            pos = sm.end()
        seg = para[pos:]
        if pieces and ABBR_RE.search(pieces[-1][2]):
            pieces[-1] = (pieces[-1][0], len(para), pieces[-1][2] + para[pieces[-1][1]:])
        else:
            pieces.append((pos, len(para), seg))
        for a, b, s in pieces:
            if not s.strip():
                continue
            va, vb = base + a, base + b
            ta, tb = mt.tex_span(va, vb)
            out.append({"text": s.strip(), "view_span": [va, vb], "tex_span": [ta, tb], "para": pm.start()})
    return out


def tokens(s: str) -> list[str]:
    """Word tokens. Sentinels survive because they are alphabetic."""
    return re.findall(r"[A-Za-z][A-Za-z\-']*|\d+(?:\.\d+)?", s)


STOP = set("""the a an of and or to in on for with by is are was were be been being this that these those we our it its
as at from than then thus which such can may also more most other each both same very using used use no not all any
into over under between within across per via i e g""".split())


def content_tokens(toks: list[str]) -> int:
    return sum(1 for t in toks if t.lower() not in STOP and not re.fullmatch(r"\d+(?:\.\d+)?", t) and t.lower() not in SENTINELS)


# --------------------------------------------------------------------------- references

def ref_events(doc: Doc, include_appendix: bool = True) -> list[dict]:
    """Every reference-like macro call in the document body with its speaking section and
    the section that declares the label. Multi-argument macros (\\twosecrefs{a}{b}) yield one
    event per argument."""
    out = []
    for m in REF_MACRO_RE.finditer(doc.tex):
        if not (doc.doc_start <= m.start() < doc.doc_end):
            continue
        name = m.group(1)
        if REF_BLACKLIST.search(name):
            continue
        src = doc.top_of(m.start())
        in_app = doc.appendix_start is not None and m.start() >= doc.appendix_start
        if in_app and not include_appendix:
            continue
        for arg in re.findall(r"\{([^{}]*)\}", m.group(2)):
            for lab in [x.strip() for x in arg.split(",") if x.strip()]:
                info = doc.labels.get(lab)
                out.append({"macro": name, "label": lab, "offset": m.start(), "src_section": src,
                            "src_in_appendix": in_app,
                            "dst_section": info["section"] if info else None,
                            "dst_kind": info["kind"] if info else "undefined",
                            "dst_in_appendix": (info is not None and doc.appendix_start is not None
                                                and info["offset"] >= doc.appendix_start)})
    return out


# --------------------------------------------------------------------------- textual section refs

TEXT_SECREF_RE = re.compile(r"\b(?:Section|Sections|Sec\.|Secs\.|Appendix|Appendices|App\.|§)\s*~?\s*([A-Z]?\d+(?:\.\d+)*|[A-Z])(?:\s*(?:,|and|&)\s*~?\s*([A-Z]?\d+(?:\.\d+)*|[A-Z]))*")


def text_section_refs(mt: MappedText) -> list[dict]:
    """Second parser: explicit 'Section 4' style references in prose (works on the P view,
    where \\ref calls are already sentinels, so it catches only hand-written numbers)."""
    out = []
    for m in TEXT_SECREF_RE.finditer(mt.text):
        nums = re.findall(r"[A-Z]?\d+(?:\.\d+)*|\b[A-Z]\b", m.group(0))
        out.append({"text": m.group(0), "targets": nums, "view_span": [m.start(), m.end()],
                    "tex_span": list(mt.tex_span(m.start(), m.end()))})
    return out
