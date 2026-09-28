"""PairAcc and AUROC for the baselines on benchA4S, in the layout the paper's baseline table uses.

Scores and their directions are the ones bench165 fixed, so the two benches can be read side by
side. Binoculars is lower for machine text, DetectGPT and Fast-DetectGPT and NTS are higher, and a
reviewer's Overall rating is lower for the AI paper if the reviewer sees anything at all. A missing
file is printed as absent rather than skipped, because an absent baseline is a claim about the
experiment and not a blank in the table. Pangram has no row it can fill; the account's key returns
402 Insufficient credits, as it did for bench165.
"""
import json, os, glob

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/itemsA4S.json"))["items"]
label = {i["item_id"]: i["label"] for i in items}
pair = {i["item_id"]: i["pair"] for i in items}

DETECTORS = [("Binoculars (falcon, faithful)", "baselines/binoculars_faithful.jsonl", "binoculars", -1),
             ("Binoculars (qwen pair)", "baselines/binoculars.jsonl", "binoculars", -1),
             ("DetectGPT", "baselines/detectgpt.jsonl", "detectgpt", +1),
             ("Fast-DetectGPT", "baselines/fast_detectgpt.jsonl", "fast_detectgpt", +1),
             ("NTS", "baselines/nts.jsonl", "nts", +1),
             ("Pangram", None, None, None)]
REVIEWERS = [("CycleReviewer", "b2h"), ("AI Scientist", "b3a"), ("CMU reviewer", "b3i")]


def auroc(pos, neg, d):
    if not pos or not neg:
        return None
    n = sum(1.0 if (a - b) * d > 0 else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(n / (len(pos) * len(neg)), 4)


def report(name, scores, d):
    """scores: item_id -> value."""
    by = {}
    for k, v in scores.items():
        if k in pair and v is not None:
            by.setdefault(pair[k], {})[label[k]] = float(v)
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    if not both:
        return name, None, None, 0, None
    w = sum(1 for a, h in both if (a - h) * d > 0)
    t = sum(1 for a, h in both if a == h)
    ai = [v[1] for v in by.values() if 1 in v]
    hu = [v[0] for v in by.values() if 0 in v]
    return (name, round((w + 0.5 * t) / len(both), 4), auroc(ai, hu, d), len(both),
            (w, len(both) - w - t, t))


rows = []
for name, rel, field, d in DETECTORS:
    if rel is None:
        rows.append((name, None, None, 0, "no API credit (402)"))
        continue
    f = f"{B}/results/{rel}"
    if not os.path.isfile(f):
        rows.append((name, None, None, 0, "not yet available"))
        continue
    sc = {}
    for line in open(f):
        r = json.loads(line)
        sc[r["id"]] = r.get(field)
    rows.append(report(name, sc, d))

for name, sysname in REVIEWERS:
    fs = glob.glob(f"{B}/results/reviews/{sysname}/*.json")
    if not fs:
        rows.append((name, None, None, 0, "not yet available"))
        continue
    sc = {}
    for f in fs:
        d0 = json.load(open(f))
        fin = d0.get("final")
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)):
            sc[d0["item_id"]] = v
    rows.append(report(name, sc, -1))

w = max(len(r[0]) for r in rows)
print(f"{'baseline':<{w}}  {'PairAcc':>8}  {'AUROC':>7}  {'pairs':>5}   W/L/T")
for name, pa, au, n, wlt in rows:
    if pa is None:
        print(f"{name:<{w}}  {'-':>8}  {'-':>7}  {'-':>4}   {wlt if wlt else ''}")
    else:
        print(f"{name:<{w}}  {pa:>8.4f}  {au:>7.4f}  {n:>4}   {wlt[0]}/{wlt[1]}/{wlt[2]}")
