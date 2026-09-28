"""Broken enumeration inside a figure, counted from the transcripts of both benches.

A figure that labels its parts Phase I, Phase II, Phase IV, Phase V is missing Phase III, and the
gap is visible to anyone who looks at the drawing once. The check is deterministic, so it needs no
model and can be repeated. Roman and arabic numerals both count, and a single label is not a
sequence.
"""
import json, re, sys, os
from collections import defaultdict

SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
WORDS = r"(?:phase|stage|step|module|layer|part|round|level)"
PAT = re.compile(rf"\b{WORDS}\s*[:\-]?\s*(\d{{1,2}}|[ivx]{{1,4}})\b", re.I)


def numbers(lines):
    """label word -> sorted distinct ordinals, so Phase and Step are separate sequences."""
    seq = defaultdict(set)
    for x in lines:
        for m in re.finditer(rf"\b({WORDS})\s*[:\-]?\s*(\d{{1,2}}|[ivx]{{1,4}})\b", str(x), re.I):
            word, tok = m.group(1).lower(), m.group(2).lower()
            n = int(tok) if tok.isdigit() else ROMAN.get(tok)
            if n:
                seq[word].add(n)
    return {w: sorted(v) for w, v in seq.items()}


def gap(seq):
    """True when a sequence of three or more labels skips an ordinal inside its own range."""
    for w, v in seq.items():
        if len(v) >= 3 and v[-1] - v[0] + 1 > len(v):
            return True, f"{w} {v}"
    return False, ""


def run(path, label):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for r in rows:
        lines = [str(x) for x in ((r.get("result") or {}).get("lines") or [])]
        seq = numbers(lines)
        g, why = gap(seq)
        out.setdefault(r["corpus"], []).append((g, why, r["id"]))
    print(f"=== {label}")
    for c, v in out.items():
        n = sum(1 for g, _, _ in v if g)
        have = sum(1 for g, w, i in v if numbers([]) or True and any(True for _ in [1]))
        seqed = sum(1 for r in rows if r["corpus"] == c
                    and any(len(x) >= 3 for x in numbers([str(y) for y in ((r.get("result") or {}).get("lines") or [])]).values()))
        print(f"  {c}: figures {len(v)} | numbered sequence of 3+ {seqed} | skipped {n} ({n/len(v):.1%})")
        for g, why, i in v:
            if g:
                print(f"      {i}: {why}")


run(f"{B}/results/slop/fig_exposition/transcripts.jsonl", "benchA4S")
p165 = f"{SLOP}/Artifacts/fig_exposition/results/transcripts.jsonl"
if os.path.exists(p165):
    run(p165, "FARS bench165")
