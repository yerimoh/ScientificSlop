"""Build benchmark items and views for the Agents4Science pairs, in the bench165 format.

Same rules as bench165: body = expanded tex up to the first of \\appendix / Acknowledgement /
bibliography, identical on both sides; views body.tex (structure) and body.txt (prose for text
detectors). The human side is always the anchor's LaTeX e-print, read exactly as FARS's human side
is. The AI side reads the manuscript source when the submission shipped one (a4s_tex / a4s_arxiv,
recorded in pairs_a4s_v7.json), and otherwise the TeX-shaped document scripts/pdf_to_tex.py rebuilds
from the PDF, so every paper passes through one reader. ai_source records which of the two a paper
used, because a rebuilt document recovers fewer \\ref and \\label than a real source.

Items carry sim_tier, pool and the A4S-only fields (venue decision, reviewer scores, autonomy
self-report) so analysis can stratify on them.
"""
import json, os, re, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SB = f"{ROOT}/paper/draft_v6/scislopbench"
A = f"{SB}/data/Agents4Science"
B = f"{SB}/benchA4S"
sys.path.insert(0, f"{SB}/data/scripts")
from slopbench_lib import flatten, flatten_dir, split_body, metrics, find_root_tex  # noqa: E402

SRC = {"human_refs/eprints/": f"{ROOT}/ana/reference/data/human_refs/eprints",
       "a4s/eprints_new/": f"{A}/eprints_new",
       "eprints_new/": f"{SB}/data/eprints_new",
       "pairs165/eprints_new/": f"{SB}/data/pairs165_0911/eprints_new"}


def prose_view(tex):
    """Identical to bench165's prose view so text detectors see the same surface on both benches."""
    s = tex
    s = re.sub(r'\\begin\{(figure|table|algorithm|algorithmic|tikzpicture)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'\\caption\s*(?:\[[^\]]*\])?\s*\{', ' [CAPTION] ', s)
    s = re.sub(r'\$\$.*?\$\$', ' [MATH] ', s, flags=re.S)
    s = re.sub(r'\$[^$\n]{0,400}\$', ' [MATH] ', s)
    s = re.sub(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{[^}]*\}', ' [CITE] ', s)
    s = re.sub(r'\\(?:auto|c|C|eq)?ref\*?\{[^}]*\}', ' [REF] ', s)
    s = re.sub(r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}', r'\n\n## \2\n', s)
    s = re.sub(r'\\(label|includegraphics|url|href)\s*(\[[^\]]*\])?\s*\{[^}]*\}', ' ', s)
    s = re.sub(r'\\begin\{[^}]*\}|\\end\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?', ' ', s)
    s = re.sub(r'[{}~]', ' ', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def hu_dir(h):
    for k, base in SRC.items():
        if h["tex_source"].startswith(k):
            return os.path.join(base, h["tex_source"][len(k):])
    return os.path.join(f"{SB}/data", h["tex_source"])


def main():
    d = json.load(open(f"{A}/pairs_a4s.json"))
    # The submissions that shipped LaTeX, taken straight from the root chooser rather than from the
    # v7 pair list, which stopped at the eleven that had a pair back then. A root counts only when
    # the PDF's own sentences are found in it, so a stale or partial source is not read as the paper.
    tex_index = json.load(open(f"{A}/cache/a4s_tex_index.json"))
    real = {c: os.path.join(A, v["root"]) for c, v in tex_index.items()
            if v["pdf_sentence_coverage"] >= 0.85 and v["metrics"]["body_words"] >= 800}
    items, fails = [], []
    for p in d["pairs"]:
        code, h = p["code"], p.get("human_primary")
        if not h:
            continue
        try:
            ai_main = real.get(code) or f"{B}/ai_tex/{code}/main.tex"
            ai_source = "shipped_tex" if code in real else "rebuilt_from_pdf"
            body, _, _ = split_body(flatten(ai_main))
            aid = f"AI_{code}"
            os.makedirs(f"{B}/views/{aid}", exist_ok=True)
            open(f"{B}/views/{aid}/body.tex", "w").write(body)
            open(f"{B}/views/{aid}/body.txt", "w").write(prose_view(body))
            m1 = metrics(flatten(ai_main))
            items.append(dict(item_id=aid, pair=code, label=1, sim_tier=h.get("sim_tier"),
                              pool=p.get("pool"), body_words=m1["body_words"],
                              tex_dir=os.path.dirname(ai_main), tex_root=os.path.basename(ai_main),
                              ai_source=ai_source, pdf=p["ai"]["pdf"],
                              a4s_status=p["a4s_status"], contribution_type=p["a4s_type"],
                              scores=p.get("a4s_scores"), autonomy=p.get("a4s_autonomy")))

            base = hu_dir(h)
            r2, full2 = flatten_dir(base)
            body2, _, _ = split_body(full2)
            hid = f"HU_{h['arxiv']}"
            os.makedirs(f"{B}/views/{hid}", exist_ok=True)
            open(f"{B}/views/{hid}/body.tex", "w").write(body2)
            open(f"{B}/views/{hid}/body.txt", "w").write(prose_view(body2))
            m2 = metrics(full2)
            items.append(dict(item_id=hid, pair=code, label=0, sim_tier=h.get("sim_tier"),
                              pool=p.get("pool"), arxiv=h["arxiv"], venue=h.get("venue"),
                              iclr_rating=h.get("iclr_rating"), body_words=m2["body_words"],
                              contribution_type=p["a4s_type"],
                              tex_dir=base, tex_root=os.path.relpath(r2, base)))
        except Exception as e:
            fails.append((code, f"{type(e).__name__}: {e}"[:140]))

    out = dict(generated="2026-09-18", pairs_manifest="data/Agents4Science/pairs_a4s.json",
               body_rule="expanded tex to the first of appendix / acknowledgement / bibliography",
               n_pairs=len(items) // 2, items=items, build_failures=fails)
    json.dump(out, open(f"{B}/itemsA4S.json", "w"), indent=1, ensure_ascii=False)

    os.makedirs(f"{B}/scripts", exist_ok=True)
    with open(f"{B}/scripts/texts.jsonl", "w") as w:
        for it in items:
            t = open(f"{B}/views/{it['item_id']}/body.txt").read()
            w.write(json.dumps(dict(id=it["item_id"], label=it["label"], year=0,
                                    source="benchA4S_AI" if it["label"] == 1 else "benchA4S_HU",
                                    text=t), ensure_ascii=False) + "\n")
    print(f"pairs {len(items)//2} | items {len(items)} | build failures {len(fails)}")
    for c, e in fails:
        print("  ", c, e)


if __name__ == "__main__":
    main()
