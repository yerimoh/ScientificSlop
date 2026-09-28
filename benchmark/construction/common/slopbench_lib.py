"""Shared helpers: document-order TeX flattening, body/appendix split, structural metrics."""
import os, re, glob, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
EPRINTS = f"{ROOT}/ana/reference/data/human_refs/eprints"
FARS = f"{ROOT}/fars/papers"

INPUT_RE = re.compile(r'\\(?:input|include|subfile|subfileinclude)\s*\{([^}]+)\}')
IMPORT_RE = re.compile(r'\\(?:import|subimport)\s*\{([^}]+)\}\s*\{([^}]+)\}')


def strip_comments(s):
    out = []
    for line in s.split("\n"):
        # remove unescaped % to end of line
        i = 0; res = []
        while i < len(line):
            c = line[i]
            if c == "\\" and i + 1 < len(line):
                res.append(line[i:i+2]); i += 2; continue
            if c == "%":
                break
            res.append(c); i += 1
        out.append("".join(res))
    return "\n".join(out)


def find_root_tex(d):
    cands = []
    for f in glob.glob(os.path.join(d, "**", "*.tex"), recursive=True):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if re.search(r'\\documentclass', s) and re.search(r'\\begin\{document\}', s):
            cands.append((f, len(s)))
    if not cands:
        return None
    # prefer main.tex / iclr*conference.tex / shortest path depth, then largest
    def key(x):
        f, n = x
        base = os.path.basename(f).lower()
        pri = 0 if base in ("main.tex",) else (1 if "conference" in base or "paper" in base else 2)
        return (f.count(os.sep), pri, -n)
    cands.sort(key=key)
    return cands[0][0]


def _resolve(base_dir, root_dir, name):
    name = name.strip()
    for cand in (name, name + ".tex"):
        for d in (root_dir, base_dir):
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                return p
    return None


def flatten(root_tex, _seen=None, depth=0, root_dir=None):
    """Recursively expand \\input at their positions, comments stripped.
    Paths resolve against the true root directory first (TeX semantics), then the including file's dir."""
    if _seen is None:
        _seen = set()
    if root_tex in _seen or depth > 12:
        return ""
    _seen.add(root_tex)
    s = open(root_tex, encoding="utf-8", errors="ignore").read()
    s = strip_comments(s)
    base = os.path.dirname(root_tex)
    if root_dir is None:
        root_dir = base

    def sub_import(m):
        d = os.path.join(base, m.group(1))
        p = _resolve(d, root_dir, m.group(2))
        return flatten(p, _seen, depth + 1, root_dir) if p else ""
    s = IMPORT_RE.sub(sub_import, s)

    def sub_input(m):
        p = _resolve(base, root_dir, m.group(1))
        if p is None:
            return ""
        return "\n" + flatten(p, _seen, depth + 1, root_dir) + "\n"
    s = INPUT_RE.sub(sub_input, s)
    return s


def flatten_dir(d):
    r = find_root_tex(d)
    if r is None:
        return None, None
    return r, flatten(r)


BODY_END_RE = re.compile(
    r'\\appendix\b|\\section\*?\{\s*Acknowledg|\\begin\{thebibliography\}|\\bibliography\s*\{|\\printbibliography|\\end\{document\}',
    re.I)


def split_body(full):
    m = re.search(r'\\begin\{document\}', full)
    doc = full[m.end():] if m else full
    m2 = BODY_END_RE.search(doc)
    body = doc[:m2.start()] if m2 else doc
    rest = doc[m2.start():] if m2 else ""
    has_appendix = bool(re.search(r'\\appendix\b', doc))
    return body, rest, has_appendix


def prose_words(tex):
    """Approximate word count of prose (math, floats, macros removed)."""
    s = tex
    s = re.sub(r'\\\\\s*\[[^\]]*\]', ' ', s)          # line breaks with spacing \\[2pt]
    s = s.replace('\\$', ' ').replace('\\%', ' ')          # escaped dollar / percent
    s = re.sub(r'\\begin\{(figure|table|algorithm|algorithmic|tikzpicture|tabular|equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'\$\$.*?\$\$', ' ', s, flags=re.S)
    s = re.sub(r'\$[^$\n]{0,400}\$', ' ', s)
    s = re.sub(r'(?<!\\)\\\[.*?\\\]', ' ', s, flags=re.S)
    s = re.sub(r'\\(cite\w*|ref|autoref|cref|Cref|eqref|label|includegraphics|url|href)\s*(\[[^\]]*\])?\s*\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?', ' ', s)
    s = re.sub(r'[{}\[\]~]', ' ', s)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", s)
    return len(words)


def metrics(full):
    body, rest, has_app = split_body(full)
    sec = re.findall(r'\\section\*?\{([^}]*)\}', body)
    subsec = re.findall(r'\\subsection\*?\{', body)
    figs = len(re.findall(r'\\begin\{figure\*?\}', body))
    tabs = len(re.findall(r'\\begin\{table\*?\}', body))
    algs = len(re.findall(r'\\begin\{algorithm\*?\}', body))
    eqs = len(re.findall(r'\\begin\{(equation|align|gather|multline)\*?\}', body)) + len(re.findall(r'\\\[', body))
    cites = re.findall(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}', body)
    keys = set()
    for c in cites:
        for k in c.split(","):
            k = k.strip()
            if k:
                keys.add(k)
    refs = len(re.findall(r'\\(?:auto|c|C|eq)?ref\*?\{', body))
    app_words = prose_words(rest) if has_app else 0
    rw = any(re.search(r'related|background|prior work|literature', s, re.I) for s in sec)
    gh = sorted(set(re.findall(r'github\.com/([\w.\-]+/[\w.\-]+)', full)))
    tmpl = re.findall(r'\\usepackage(?:\[[^\]]*\])?\{([^}]*(?:iclr|neurips|icml|colm|acl|emnlp|naacl|aaai|cvpr|analemma)[^}]*)\}', full, re.I)
    return dict(
        github=gh[:5], template=tmpl[:3],
        body_words=prose_words(body), appendix_words=app_words, has_appendix=has_app,
        n_sections=len(sec), section_titles=[re.sub(r'\s+', ' ', s).strip() for s in sec],
        n_subsections=len(subsec), n_figures=figs, n_tables=tabs, n_algorithms=algs,
        n_equations=eqs, n_cite_cmds=len(cites), n_unique_cites=len(keys), n_internal_refs=refs,
        has_related_work=rw,
    )


def fars_dir(code):
    ds = glob.glob(f"{FARS}/{code}_*")
    return ds[0] if ds else None


def fars_tex_root(code):
    d = fars_dir(code)
    p = os.path.join(d, "code", "writing", "paper", "main.tex")
    return p if os.path.isfile(p) else None
