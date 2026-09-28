"""Build the full-corpus benchmark items and views from pairs165_draft.json (144 pairs).

Same rules as bench5: body = expanded tex up to the first of \\appendix / Acknowledgement /
bibliography, identical on both sides; views body.tex (structure) and body.txt (prose for
text detectors). Items carry sim_tier and pool so analysis can filter to tier >= 2.
Frozen bench5 pairs are included (rebuilt here so bench165 is self-contained).
"""
import json, os, re, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SB = f"{ROOT}/paper/draft_v6/scislopbench"
B = f"{SB}/bench165"
sys.path.insert(0, f"{SB}/data/scripts")
from slopbench_lib import flatten, flatten_dir, split_body, metrics, fars_dir, fars_tex_root  # noqa: E402


def prose_view(tex):
    s = tex
    s = re.sub(r'\\begin\{(figure|table|algorithm|algorithmic|tikzpicture)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'\\caption\s*(?:\[[^\]]*\])?\s*\{', ' [CAPTION] ', s)
    s = re.sub(r'\$\$.*?\$\$', ' [MATH] ', s, flags=re.S); s = re.sub(r'\$[^$\n]{0,400}\$', ' [MATH] ', s)
    s = re.sub(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{[^}]*\}', ' [CITE] ', s)
    s = re.sub(r'\\(?:auto|c|C|eq)?ref\*?\{[^}]*\}', ' [REF] ', s)
    s = re.sub(r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}', r'\n\n## \2\n', s)
    s = re.sub(r'\\(label|includegraphics|url|href)\s*(\[[^\]]*\])?\s*\{[^}]*\}', ' ', s)
    s = re.sub(r'\\begin\{[^}]*\}|\\end\{[^}]*\}', ' ', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?', ' ', s); s = re.sub(r'[{}~]', ' ', s)
    s = re.sub(r'[ \t]+', ' ', s); s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def hu_dir(h):
    ts = h["tex_source"]
    if ts.startswith("human_refs"):
        return f"{ROOT}/ana/reference/data/human_refs/eprints/{h['arxiv']}"
    if ts.startswith("pairs165"):
        return f"{SB}/data/pairs165_0911/eprints_new/{h['arxiv']}"
    if ts.startswith("eprints_new"):
        return f"{SB}/data/eprints_new/{h['arxiv']}"
    return f"{SB}/data/{ts}"


d = json.load(open(f"{SB}/data/pairs165_0911/pairs165_draft.json"))
pairs = [p for p in d["pairs"] if p.get("human_primary") and p["code"] != "FA0007"]  # FA0007 ships no LaTeX
items, fails = [], []
for p in pairs:
    code = p["code"]; h = p["human_primary"]
    try:
        root = fars_tex_root(code)
        body, _, _ = split_body(flatten(root))
        aid = f"AI_{code}"
        os.makedirs(f"{B}/views/{aid}", exist_ok=True)
        open(f"{B}/views/{aid}/body.tex", "w").write(body)
        open(f"{B}/views/{aid}/body.txt", "w").write(prose_view(body))
        items.append(dict(item_id=aid, pair=code, label=1, sim_tier=h.get("sim_tier"), pool=h.get("pool", "cited"),
                          body_words=p["ai"]["body_words"]))
        base = hu_dir(h)
        r2, full2 = flatten_dir(base)
        body2, _, _ = split_body(full2)
        hid = f"HU_{h['arxiv']}"
        os.makedirs(f"{B}/views/{hid}", exist_ok=True)
        open(f"{B}/views/{hid}/body.tex", "w").write(body2)
        open(f"{B}/views/{hid}/body.txt", "w").write(prose_view(body2))
        m2 = metrics(full2)
        items.append(dict(item_id=hid, pair=code, label=0, sim_tier=h.get("sim_tier"), pool=h.get("pool", "cited"),
                          arxiv=h["arxiv"], venue=h.get("venue"), iclr_rating=h.get("iclr_rating"),
                          body_words=m2["body_words"], tex_dir=base, tex_root=os.path.relpath(r2, base)))
    except Exception as e:
        fails.append((code, str(e)[:120]))

json.dump(dict(generated="2026-09-11", pairs_manifest="data/pairs165_0911/pairs165_draft.json",
               body_rule="same as bench5", n_pairs=len(pairs), items=items, build_failures=fails),
          open(f"{B}/items165.json", "w"), indent=1, ensure_ascii=False)

os.makedirs(f"{B}/scripts", exist_ok=True)
with open(f"{B}/scripts/texts.jsonl", "w") as w:
    for it in items:
        t = open(f"{B}/views/{it['item_id']}/body.txt").read()
        w.write(json.dumps(dict(id=it["item_id"], label=it["label"], year=0,
                                source="bench165_AI" if it["label"] == 1 else "bench165_HU", text=t)) + "\n")
print("pairs", len(pairs), "items", len(items), "failures", fails)
