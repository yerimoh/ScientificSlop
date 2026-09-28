#!/usr/bin/env python3
"""fig_exposition  -  Expository apparatus of the text drawn into the method figure (Artifacts plane).

A research article divides the work of exposition. The caption carries the key to the figure's
notation, the experiments section carries the settings, the results section carries the outcomes and
the comparison, and the prose carries the paper's thesis. A method figure that carries these itself
needs no text around it, and the item counts how much of that apparatus one method figure holds.

Six expository kinds are checked on every readable method figure, and the score is the share the
figure internalises:

    notation_key          a key to the figure's own notation: a "Legend" header, a "colour = role"
                          line, a "solid or dashed arrow = kind of flow" line, or a bare arrow label
                          such as "Data flow" set beside a sample arrow
    comparison_arm        boxes tagged (Ours), (Proposed), (Baseline), (Prior), or arms of the
                          experiment labelled "Condition A", "Condition B"
    experimental_content  a run setting, a result value or an evaluation panel (hand-verified in
                          leak_verdicts.py, the one kind a pattern cannot decide)
    evaluative_mark       check marks and crosses stamped on boxes or lines
    thesis_box            a box headed "Key insight", "Key idea", "Why it works", "Takeaway",
                          "Novelty", "Key contribution"
    enumerated_stage      two or more "Stage N" / "Step N" / "Phase N" headers

Excluded by construction, with the reason in the definition document: panel letters and caption text
(both corpora carry them), an in-figure title (the human figure is a crop of a rendered page and takes
the running head with it, so the two sides are not read the same way), declarative sentences inside
boxes (human figures carry more of them), bare Input and Output labels and Problem / Goal / Solution
headers (ordinary diagram labels), nominalised box titles such as "Reward Computation" (a naming habit),
and the words Standard, Vanilla, Comparison, vs (they overlap comparison_arm and add human hits).
Text mass is recorded as an observation and never enters the score.

Both corpora are read by one whole-figure transcription call (PROMPT_transcribe.txt), and the patterns
run on the transcript lines after the running heads of the rendered page are dropped.

Usage: python3 measure.py [--corpus AI|HU|all] [--only IDS] [--limit N] [--transcribe]
       -> ../results/{elements.jsonl,papers.jsonl,summary.json}
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from records import instance, write_jsonl, rate, VIOLATION, CONSISTENT

import leak_verdicts as LEAK

RESULTS = os.path.join(HERE, "..", "results")
CENSUS = os.path.abspath(os.path.join(HERE, "..", "..", "candidte", "_census_0914"))
CHECKER = "fig_exposition_v1"

# Transcription caches, read in order; a later file overrides an earlier one for the same figure.
# The two census files are the run that produced this item's numbers; the third is this item's own.
CACHES = [os.path.join(CENSUS, "pairs165_full", "transcripts.jsonl"),
          os.path.join(CENSUS, "pairs165_full", "transcripts_fix.jsonl"),
          os.path.join(RESULTS, "transcripts.jsonl")]
MANIFEST = os.path.join(CENSUS, "pairs165_full", "data", "pairs165_manifest.json")
CROPS = os.path.join(CENSUS, "pairs165_full", "data", "pairs165_crops")
# Which human paper has a method figure, and which figure it is, was decided by hand from the caption
# list of every paper; "none" records a paper that draws no method figure and is NA, never clean.
HU_FIGURE = os.path.join(CENSUS, "pairs165_hu_overrides.json")

# A rendered page puts its running head into the crop; it is not part of the figure.
HEADER = re.compile(r"published as a conference paper|preprint|under review|arxiv:\d|^\d+$", re.I)

KIND_PAT = {
    "notation_key": re.compile(
        r"^\s*legend\b"                                              # a legend header
        r"|\b(solid|dashed|dotted)\s+(arrow|line)s?\s*[:=]"          # line style = kind of flow
        r"|\b(light\s+)?(blue|green|orange|yellow|red|purple|gray|grey|pink)\s*(fill|box(es)?)?\s*[:=]\s*\w"
        r"|=\s*(data flow|main flow|feedback|repair|conditional)"    # role named after an equals sign
        r"|^\s*(→\s*)?(\w+\s+){0,2}flow\s*$", re.I),                 # a bare arrow label
    "comparison_arm": re.compile(
        r"\((ours|proposed|baseline|prior|existing)\)|\b(ours|baseline)\s*[:—-]|\bcondition\s+[A-D]\b", re.I),
    "evaluative_mark": re.compile(r"[✓✗✅❌✔✘☑☒]"),
    "thesis_box": re.compile(
        r"\b(key (insight|idea|contribution|innovation|advantage)|why (it|this) works|takeaway|novelty)\b", re.I),
    # 0920. The one-sentence slot right under the title stating what the method does and what it achieves. The diagram
    # has taken over the caption's job, and as a one-liner under a headline it is the same device as a "Key insight" box. On the 165 pairs, humans 0/111,
    # AI only rises by one from 21/143 to 22/143, so instead of a new kind we fold it into this one.
    "thesis_gloss_TOP3": re.compile(
        r"\b(utiliz\w+|leverag\w+|employ\w+|combin\w+|appl\w+|integrat\w+)\b.{0,120}?\b(to|for)\s+\w+"
        r"|\b(enabl\w+|ensur\w+|mitigat\w+|achiev\w+|improv\w+)\b.{0,80}$", re.I),
    "enumerated_stage": re.compile(r"^\s*(stage|step|phase)\s*[0-9ivx]+\b", re.I),
}
KINDS = ["notation_key", "comparison_arm", "experimental_content",
         "evaluative_mark", "thesis_box", "enumerated_stage"]
MIN_STAGES = 2          # one "Step 1" is a label; a sequence of them is a build order
GLOSS_TOP, GLOSS_MIN = 3, 60      # within three lines under the title, long enough to count as a sentence
GLOSS_END = re.compile(r"[.!?]\s*$")
N_KINDS = len(KINDS)


def load_cache() -> dict:
    """key -> transcript lines, where key is AI_<code> or HU_<arxiv>."""
    tr = {}
    for path in CACHES:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            lines = (r.get("result") or {}).get("lines") or []
            tr[r["key"]] = [str(x) for x in lines]
    return tr


def figures() -> list[dict]:
    """One row per method figure of the pair corpus, both sides, with its id and image path.

    A paper with no method figure is not listed at all; it is not applicable to this item, and a
    figure it does not draw must never be scored as clean.
    """
    hu_fig = json.load(open(HU_FIGURE))
    out = []
    for m in json.load(open(MANIFEST)):
        if m.get("ai_png"):
            out.append({"corpus": "AI", "id": m["code"], "key": f"AI_{m['code']}",
                        "pair": m["arxiv"], "path": m["ai_png"]})
        crop = os.path.join(CROPS, f"HU_{m['arxiv']}.png")
        if hu_fig.get(m["arxiv"]) != "none" and os.path.exists(crop):
            out.append({"corpus": "HU", "id": m["arxiv"], "key": f"HU_{m['arxiv']}",
                        "pair": m["code"], "path": crop})
    return out


def read_kinds(lines: list[str], pid: str) -> tuple[dict, int]:
    """Which expository kinds this figure internalises, with the line that shows each one."""
    body = [x for x in lines if not HEADER.search(x)]
    found = {}
    for kind, pat in KIND_PAT.items():
        if kind == "thesis_gloss_TOP3":
            continue
        hits = [x for x in body if pat.search(x)]
        if kind == "enumerated_stage" and len(hits) < MIN_STAGES:
            continue
        if hits:
            found[kind] = hits
    # The one-sentence slot under the title that summarises the method. Position, length and sentence ending must all match, so a
    # regex alone cannot decide it. When it fires, it is merged into thesis_box. The number of kinds stays at six.
    gloss = [x for x in body[:GLOSS_TOP] if len(x) >= GLOSS_MIN and GLOSS_END.search(x) and "?" not in x
             and KIND_PAT["thesis_gloss_TOP3"].search(x)]
    if gloss:
        found.setdefault("thesis_box", []).extend(g for g in gloss if g not in found.get("thesis_box", []))
    if pid in LEAK.CLEAR:
        found["experimental_content"] = [LEAK.CLEAR[pid]]
    return found, sum(len(x.split()) for x in body)


def measure(fig: dict, lines: list[str]):
    found, words = read_kinds(lines, fig["id"])
    insts = []
    for k, kind in enumerate(KINDS, 1):
        hit = found.get(kind)
        insts.append(instance(
            fig["id"], "fig_exposition", k, VIOLATION if hit else CONSISTENT,
            {"kind": "figure", "figure": os.path.basename(fig["path"]), "element": kind},
            {"kind": "text", "excerpt": (hit[0][:120] if hit else None),
             "n_lines_matched": len(hit) if hit else 0},
            checker=CHECKER, verdict="internalised" if hit else "absent",
            decided_by="hand" if kind == "experimental_content" else "pattern"))
    row = {"corpus": fig["corpus"], "id": fig["id"], "pair": fig["pair"], "applicable": True,
           "n_lines": len(lines), "words": words,
           "kinds": sorted(found), "n_kinds": len(found),
           "slop_score": round(len(found) / N_KINDS, 4),
           "slop_numerator": len(found), "slop_denominator": N_KINDS,
           "coverage": 1.0, "label": "observational"}
    return row, insts


def transcribe(missing: list[dict]) -> None:
    """Read uncached figures with the same instrument as the cached ones. Needs a GPU."""
    sys.path.insert(0, CENSUS)
    import pairs165_retranscribe as RT
    from pairs165_transcribe import load
    prompt = open(os.path.join(HERE, "PROMPT_transcribe.txt")).read().strip()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "transcripts.jsonl"), "a", encoding="utf-8") as fo:
        for f in missing:
            answer, ntok = RT.ask_full(load(f["path"]), prompt)
            lines, how = RT.parse(answer)
            fo.write(json.dumps({**f, "result": {"lines": lines}, "parse": how,
                                 "n_tokens": ntok, "full_raw": answer}, ensure_ascii=False) + "\n")
            fo.flush()
            print(f["key"], how, len(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all", choices=["AI", "HU", "all"])
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transcribe", action="store_true",
                    help="read figures missing from the cache with the vision model (GPU)")
    a = ap.parse_args()

    figs = figures()
    if a.corpus != "all":
        figs = [f for f in figs if f["corpus"] == a.corpus]
    if a.only:
        ids = set(a.only.split(","))
        figs = [f for f in figs if f["id"] in ids]
    if a.limit:
        figs = figs[:a.limit]

    tr = load_cache()
    missing = [f for f in figs if f["key"] not in tr]
    if missing and a.transcribe:
        transcribe(missing)
        tr = load_cache()
        missing = [f for f in figs if f["key"] not in tr]

    rows, insts = [], []
    for f in figs:
        if f["key"] not in tr:
            rows.append({"corpus": f["corpus"], "id": f["id"], "pair": f["pair"], "applicable": False,
                         "n_kinds": None, "slop_score": None, "slop_numerator": None,
                         "slop_denominator": None, "coverage": None, "label": "NA",
                         "na_reason": "figure not transcribed"})
            continue
        r, ii = measure(f, tr[f["key"]])
        rows.append(r)
        insts.extend(ii)
    write_jsonl(os.path.join(RESULTS, "papers.jsonl"), rows)
    write_jsonl(os.path.join(RESULTS, "elements.jsonl"), insts)

    def agg(cor):
        rs = [r for r in rows if r["corpus"] == cor and r["applicable"]]
        if not rs:
            return {"figures": 0}
        return {"figures": len(rs),
                "by_kind": {k: {"n": sum(1 for r in rs if k in r["kinds"]),
                                "rate": rate(sum(1 for r in rs if k in r["kinds"]), len(rs))} for k in KINDS},
                "any_kind": sum(1 for r in rs if r["n_kinds"] > 0),
                "n_kinds_distribution": dict(sorted(Counter(r["n_kinds"] for r in rs).items())),
                "mean_slop_score": round(statistics.mean(r["slop_score"] for r in rs), 4),
                "median_words": statistics.median(r["words"] for r in rs)}

    byid = {(r["corpus"], r["id"]): r for r in rows if r["applicable"]}
    pairs = [(byid[("AI", r["id"])], byid[("HU", r["pair"])]) for r in rows
             if r["corpus"] == "AI" and r["applicable"] and ("HU", r["pair"]) in byid]
    wins = sum(1 for ai, hu in pairs if ai["n_kinds"] > hu["n_kinds"])
    losses = sum(1 for ai, hu in pairs if ai["n_kinds"] < hu["n_kinds"])
    summary = {"checker": CHECKER, "n_figures": len(rows),
               "not_transcribed": sum(1 for r in rows if not r["applicable"]),
               "by_corpus": {c: agg(c) for c in ("AI", "HU")},
               "pairs": {"n": len(pairs), "ai_higher": wins, "human_higher": losses,
                         "tied": len(pairs) - wins - losses,
                         "pair_accuracy": rate(2 * wins + (len(pairs) - wins - losses), 2 * len(pairs))},
               "hand_decided_kind": {"name": "experimental_content", "clear_AI": len(LEAK.CLEAR_AI),
                                     "clear_HU": len(LEAK.CLEAR_HU), "borderline_AI": len(LEAK.BORDER_AI),
                                     "false_AI": len(LEAK.FALSE_AI)},
               "note": "observational; no paper label. Text is the only channel read, so every rate is a "
                       "lower bound. Report the human rate beside the AI rate and the per-kind table beside "
                       "the score. The comparison of an AI figure with its pair partner is the only "
                       "comparison the crops support."}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
