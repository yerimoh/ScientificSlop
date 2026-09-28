"""Step 1 for the 165-pair build: index every FARS paper.

For each fars/papers/FA????_* directory this records
  title, abstract (meta.json), structural metrics of the flattened main.tex (slopbench_lib.metrics),
  cite_roles: bib key -> {normalized section: count} from the flattened tex,
  bib map: bib key -> {title, arxiv} (arxiv scraped from eprint/abs/volume fields).
Output: cache/fars_index.json
"""
import os, re, sys, json, glob

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/pairs165_0911"
sys.path.insert(0, f"{DATA}/scripts")
from slopbench_lib import flatten, metrics, strip_comments

SEC_NORM = [
    (r"related|background|prior work|literature|preliminar", "related_work"),
    (r"introduction", "introduction"),
    (r"method|approach|framework|model|algorithm|proposed|design|system", "method"),
    (r"experiment|evaluation|result|empirical|analysis|study|ablation|benchmark", "experiments"),
    (r"conclusion|discussion|limitation|future", "conclusion"),
]

def norm_sec(name):
    n = name.lower()
    for pat, lab in SEC_NORM:
        if re.search(pat, n):
            return lab
    return "other"

CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
AID_RE = re.compile(r"(?:abs/|eprint\s*=\s*\{\s*|arxiv\.org/(?:abs|pdf)/|arXiv:)(\d{4}\.\d{4,5})", re.I)


def parse_bib(path):
    """key -> {title, arxiv}"""
    s = open(path, encoding="utf-8", errors="ignore").read()
    out = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", s):
        key = m.group(1)
        nxt = s.find("\n@", m.end())
        chunk = s[m.start(): nxt if nxt != -1 else len(s)]
        tm = re.search(r"\btitle\s*=\s*[\{\"]+(.+?)[\}\"]+\s*,?\s*\n", chunk, re.S)
        title = re.sub(r"\s+", " ", re.sub(r"[{}\\]", "", tm.group(1))).strip() if tm else ""
        am = AID_RE.search(chunk)
        out[key] = dict(title=title, arxiv=am.group(1) if am else None)
    return out


def cite_roles(full):
    body = full
    m = re.search(r"\\begin\{document\}", full)
    if m:
        body = full[m.end():]
    parts = re.split(r"\\section\*?\{([^}]*)\}", body)
    roles = {}
    # parts[0] is pre-first-section (abstract); label it introduction-ish "other"
    segs = [("other", parts[0])] + [(norm_sec(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    for lab, seg in segs:
        for cm in CITE_RE.finditer(seg):
            for k in cm.group(1).split(","):
                k = k.strip()
                if k:
                    roles.setdefault(k, {}).setdefault(lab, 0)
                    roles[k][lab] += 1
    return roles


def main():
    idx = {}
    dirs = sorted(glob.glob(f"{ROOT}/fars/papers/FA*"))
    for d in dirs:
        code = os.path.basename(d).split("_")[0]
        meta = json.load(open(os.path.join(d, "meta.json")))
        rec = dict(code=code, dir=os.path.relpath(d, ROOT), title=meta.get("title", ""),
                   abstract=(meta.get("abstract") or "")[:4000])
        tex = os.path.join(d, "code", "writing", "paper", "main.tex")
        if os.path.isfile(tex):
            full = flatten(tex)
            rec["metrics"] = metrics(full)
            rec["cite_roles"] = cite_roles(full)
        else:
            rec["metrics"] = None
            rec["cite_roles"] = {}
            rec["note"] = "no main.tex"
        bib = os.path.join(d, "code", "writing", "paper", "analemma.bib")
        rec["bib"] = parse_bib(bib) if os.path.isfile(bib) else {}
        idx[code] = rec
        print(code, rec["title"][:60], "| cited keys:", len(rec["cite_roles"]), "bib:", len(rec["bib"]),
              "| body:", rec["metrics"]["body_words"] if rec["metrics"] else None)
    json.dump(idx, open(f"{OUTD}/cache/fars_index.json", "w"), ensure_ascii=False)
    print("total", len(idx))


if __name__ == "__main__":
    main()
