"""Corpus registry shared by every slop checker.

AI  = FARS papers with a LaTeX source (165 of 166 directories; FA0007 ships no tex).
HU  = anchor arXiv sources matched 1:1 to FARS papers (pairs.json), stored flat in
      Baseline_Sandbagging/results/_human_tex_full/<arxiv>/.

Both sides are read through the same loader (views.load_doc) so that measurement is
symmetric. Nothing here depends on an LLM.
"""
from __future__ import annotations
import glob
import json
import os
import re

ROOT = os.environ.get("SCISLOP_ROOT", ".")
FARS = f"{ROOT}/fars/papers"
PAIRS = f"{ROOT}/fars/analysis/05_anchor_pairs/pairs.json"
HU_TEX = f"{ROOT}/Mold/Content_Mold/Baseline_Sandbagging/results/_human_tex_full"
VIZ_RESULTS = f"{ROOT}/artifact-ai2science/Cross-output_Mold/Visualization/results"
SLOP_ROOT = f"{ROOT}/paper/draft_v6/slop"

# Papers excluded from every checker, with the reason recorded once here.
EXCLUDE = {
    "FA0007": "no LaTeX source in the FARS dump (paper.pdf and paper.txt only)",
}


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def pairs() -> list[dict]:
    return json.load(open(PAIRS))


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def ai_papers() -> list[dict]:
    """One record per FARS paper that has a LaTeX main file."""
    out = []
    for d in sorted(glob.glob(f"{FARS}/FA*")):
        code = os.path.basename(d)[:6]
        if code in EXCLUDE:
            continue
        main = f"{d}/code/writing/paper/main.tex"
        if not os.path.exists(main):
            continue
        out.append({
            "corpus": "AI",
            "id": code,
            "root": d,
            "main_tex": main,
            "paper_dir": f"{d}/code/writing/paper",
            "exp_dir": f"{d}/code/exp",
            "diagram": _first_existing([f"{d}/code/writing/method_diagrams/framework_overview.png",
                                        f"{d}/code/writing/method_diagrams/framework_overview.jpeg"]),
        })
    return out


def hu_main_tex(arxiv_dir: str) -> str | None:
    """Pick the main file of a flat arXiv source dump.

    Prefer a file that has \\documentclass and \\begin{document}; if several qualify,
    take the one with the most \\input calls, then the longest.
    """
    files = glob.glob(f"{arxiv_dir}/*.tex")
    cands = []
    for f in files:
        t = read(f)
        if re.search(r"\\documentclass", t):
            score = (1 if "\\begin{document}" in t else 0,
                     len(re.findall(r"\\(?:input|include)\{", t)), len(t))
            cands.append((score, f))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def hu_papers() -> list[dict]:
    """One record per distinct anchor arXiv source that exists on disk."""
    out, seen = [], set()
    for p in pairs():
        ax = p.get("anchor_arxiv_id")
        if not ax or ax in seen:
            continue
        seen.add(ax)
        d = f"{HU_TEX}/{ax}"
        if not os.path.isdir(d):
            continue
        main = hu_main_tex(d)
        if not main:
            continue
        out.append({
            "corpus": "HU",
            "id": ax,
            "root": d,
            "main_tex": main,
            "paper_dir": d,
            "exp_dir": None,
            "diagram": None,
            "anchor_of": p["code"],
            "anchor_year": p.get("anchor_year"),
            "anchor_venue": p.get("anchor_venue"),
        })
    return out


def all_papers() -> list[dict]:
    return ai_papers() + hu_papers()


if __name__ == "__main__":
    ai, hu = ai_papers(), hu_papers()
    print(f"AI {len(ai)}  HU {len(hu)}  excluded {EXCLUDE}")
    print("AI with diagram file:", sum(1 for p in ai if p["diagram"]))
