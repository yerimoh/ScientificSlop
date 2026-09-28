"""Every score on benchA4S after the length it can be predicted from is taken out of it.

Word count alone separates the two corpora at PairAcc 0.915, so a score that rises with length
scores well without reading the paper. Each measure is regressed on log body words over all 494
documents of both corpora together, and the pair comparison is redone on the residual. A measure
that was only counting words falls to chance here; a measure that was reading the paper keeps most
of its separation. The regression is fitted on both corpora at once on purpose, so that the part of
the score that any paper of that length would get is what gets removed.
"""
import json, math, os, statistics as st

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/itemsA4S.json"))["items"]
pair = {i["item_id"]: i["pair"] for i in items}
lab = {i["item_id"]: i["label"] for i in items}
words = {}
for line in open(f"{B}/scripts/texts.jsonl"):
    r = json.loads(line)
    words[r["id"]] = len(r["text"].split())

ITEMS = [("xsec_ref", "slop_score", +1), ("evidence_gap", "slop_score", +1),
         ("citation", "slop_score", +1), ("argument_graph", "slop_score", +1),
         ("macro_redund", "slop_score", +1), ("fig_exposition", "slop_score", +1)]
DET = [("baselines/binoculars_faithful.jsonl", "binoculars", -1, "Binoculars(falcon)"),
       ("baselines/binoculars.jsonl", "binoculars", -1, "Binoculars(qwen)"),
       ("baselines/detectgpt.jsonl", "detectgpt", +1, "DetectGPT"),
       ("baselines/fast_detectgpt.jsonl", "fast_detectgpt", +1, "Fast-DetectGPT"),
       ("baselines/nts.jsonl", "nts", +1, "NTS")]


def collect_item(name, field):
    out = {}
    p = f"{B}/results/slop/{name}/papers.jsonl"
    if not os.path.isfile(p):
        return out
    for line in open(p):
        r = json.loads(line)
        key = f"AI_{r['id']}" if r["corpus"] == "AI" else f"HU_{r['id']}"
        if key in pair and r.get(field) is not None:
            out[key] = float(r[field])
    return out


def collect_det(rel, field):
    out = {}
    p = f"{B}/results/{rel}"
    if not os.path.isfile(p):
        return out
    for line in open(p):
        r = json.loads(line)
        if r["id"] in pair and r.get(field) is not None:
            out[r["id"]] = float(r[field])
    return out


def collect_rev(sysname):
    import glob
    out = {}
    for f in glob.glob(f"{B}/results/reviews/{sysname}/*.json"):
        d = json.load(open(f))
        fin = d.get("final")
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)) and d["item_id"] in pair:
            out[d["item_id"]] = float(v)
    return out


def residualise(sc):
    """score minus its least-squares fit on log words, over both corpora together."""
    ks = [k for k in sc if words.get(k)]
    if len(ks) < 20:
        return sc
    xs = [math.log(words[k]) for k in ks]
    ys = [sc[k] for k in ks]
    mx, my = st.mean(xs), st.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 0.0
    return {k: sc[k] - (my + b * (math.log(words[k]) - mx)) for k in ks}


def evaluate(sc, d):
    by = {}
    for k, v in sc.items():
        by.setdefault(pair[k], {})[lab[k]] = v
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    if not both:
        return None
    w = sum(1 for a, h in both if (a - h) * d > 0)
    t = sum(1 for a, h in both if a == h)
    ai = [v[1] for v in by.values() if 1 in v]
    hu = [v[0] for v in by.values() if 0 in v]
    au = sum(1.0 if (a - b) * d > 0 else 0.5 if a == b else 0.0
             for a in ai for b in hu) / (len(ai) * len(hu))
    return round((w + 0.5 * t) / len(both), 3), round(au, 3), len(both)


rows = []
for name, field, d in ITEMS:
    sc = collect_item(name, field)
    if sc:
        rows.append(("item " + name, evaluate(sc, d), evaluate(residualise(sc), d)))
for rel, field, d, nm in DET:
    sc = collect_det(rel, field)
    if sc:
        rows.append(("detector " + nm, evaluate(sc, d), evaluate(residualise(sc), d)))
for sysname, nm in (("b2h", "CycleReviewer"), ("b3a", "AI Scientist")):
    sc = collect_rev(sysname)
    if sc:
        rows.append(("reviewer " + nm, evaluate(sc, d=-1), evaluate(residualise(sc), d=-1)))
sc = {k: float(v) for k, v in words.items() if k in pair}
rows.append(("baseline word count", evaluate(sc, -1), evaluate(residualise(sc), -1)))

print(f"{'measure':<26}{'raw PairAcc':>13}{'AUROC':>8}{'len-adj PairAcc':>17}{'AUROC':>8}{'pairs':>6}")
for nm, a, b in sorted(rows, key=lambda r: -(r[2][0] if r[2] else 0)):
    if a is None or b is None:
        continue
    print(f"{nm:<26}{a[0]:>13.3f}{a[1]:>8.3f}{b[0]:>17.3f}{b[1]:>8.3f}{b[2]:>6}")
