"""Extract plain prose for the 1,294 ICLR 2017-2025 papers in iclr9y.pkl.
Source: data/{year}/{id}/tex  (preferred)  else data_retired/{year}/{id}/{tex,ocr.md}.
Output: texts.jsonl  {id, year, source, n_chars, text}
"""
import os, re, json, glob, sys
import pandas as pd

ROOT = os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/Evaluation/ICLR"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts.jsonl")

def find_dir(year, pid):
    for base in (f"{ROOT}/data/{year}/{pid}", f"{ROOT}/data_retired/{year}/{pid}"):
        if os.path.isdir(base):
            return base
    return None

def detex(s):
    s = re.sub(r"(?<!\\)%.*", "", s)                                  # comments
    for env in ("figure","table","tabular","equation","align","algorithm","lstlisting",
                "verbatim","thebibliography","tikzpicture","subfigure","wrapfigure","minipage"):
        s = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]{0,400}?\$", " ", s)
    s = re.sub(r"\\(cite[a-zA-Z]*|ref|eqref|label|autoref|citep|citet)\s*\{[^}]*\}", " ", s)
    s = re.sub(r"\\(section|subsection|subsubsection|paragraph|title)\*?\s*\{([^}]*)\}", r" \2. ", s)
    s = re.sub(r"\\(textbf|textit|emph|texttt|textsc|mbox|text)\s*\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = s.replace("~", " ")
    s = re.sub(r"[{}\\]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()

def demd(s):
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]{0,400}?\$", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"^\s*\|.*\|\s*$", " ", s, flags=re.M)                 # md tables
    s = re.sub(r"^#+\s*", "", s, flags=re.M)
    s = re.sub(r"[*_`>]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()

def read_paper(base):
    texdir = os.path.join(base, "tex")
    if os.path.isdir(texdir):
        files = sorted(glob.glob(os.path.join(texdir, "**", "*.tex"), recursive=True))
        if files:
            raw = []
            for f in files:
                try: raw.append(open(f, encoding="utf-8", errors="ignore").read())
                except Exception: pass
            body = "\n".join(raw)
            m = re.search(r"\\begin\{document\}", body)
            if m: body = body[m.end():]
            m = re.search(r"\\(bibliography|begin\{thebibliography\})", body)
            if m: body = body[:m.start()]
            t = detex(body)
            if len(t) > 2000: return t, "tex"
    md = os.path.join(base, "ocr.md")
    if os.path.exists(md):
        t = demd(open(md, encoding="utf-8", errors="ignore").read())
        t = re.split(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", t)[0]
        if len(t) > 2000: return t, "ocr"
    return None, None

if __name__ == "__main__":
    d = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "iclr9y.pkl"))
    n_ok = 0; miss = []
    with open(OUT, "w") as w:
        for pid, year in zip(d.pid, d.year):
            base = find_dir(year, pid)
            t, src = read_paper(base) if base else (None, None)
            if not t: miss.append((year, pid)); continue
            w.write(json.dumps({"id": pid, "year": int(year), "source": src,
                                "n_chars": len(t), "text": t}) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} / {len(d)}  -> {OUT}")
    if miss: print("missing:", len(miss), miss[:10])
