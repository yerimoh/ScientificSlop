"""benchA4S scorecard split by the two things that could be making the numbers, the strength of the
pair and where the AI manuscript came from.

match_rank 4 is a pair steps 6 and 9 assigned on a judged topic match with the same contribution
type; ranks 3 to 0 are the relaxations step 10 used to reach all 247. ai_source separates the 11
submissions that shipped LaTeX from the ones rebuilt out of the PDF, which is the only place a
missing \\ref or an unreadable table can come from.
"""
import json, os, sys

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = f"{B}/../data/Agents4Science"
items = json.load(open(f"{B}/itemsA4S.json"))["items"]
pairs = {p["code"]: p for p in json.load(open(f"{A}/pairs_a4s.json"))["pairs"]}
pair_of, meta = {}, {}
for i in items:
    key = i["pair"] if i["label"] == 1 else i.get("arxiv")
    pair_of[("AI" if i["label"] == 1 else "HU", key)] = i["pair"]
    if i["label"] == 1:
        hp = pairs[i["pair"]].get("human_primary") or {}
        meta[i["pair"]] = {"rank": hp.get("match_rank", 4), "src": i.get("ai_source"),
                           "tier": i.get("sim_tier")}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return round(sum(1.0 if a > b else 0.5 if a == b else 0.0
                     for a in pos for b in neg) / (len(pos) * len(neg)), 3)


def score(ck, keep):
    f = f"{B}/results/slop/{ck}/papers.jsonl"
    if not os.path.isfile(f):
        return None
    by = {}
    for line in open(f):
        r = json.loads(line)
        p = pair_of.get((r["corpus"], r["id"]))
        if p is None or not keep(meta.get(p, {})):
            continue
        v = r.get("slop_score")
        by.setdefault(p, {})[r["corpus"]] = None if v is None else float(v)
    both = [(v["AI"], v["HU"]) for v in by.values()
            if v.get("AI") is not None and v.get("HU") is not None]
    if not both:
        return None
    w = sum(1 for a, h in both if a > h)
    t = sum(1 for a, h in both if a == h)
    ai = [v["AI"] for v in by.values() if v.get("AI") is not None]
    hu = [v["HU"] for v in by.values() if v.get("HU") is not None]
    return round((w + 0.5 * t) / len(both), 3), auroc(ai, hu), len(both)


SPLITS = [("all 247", lambda m: True),
          ("rank 4 (verdicts agree)", lambda m: m.get("rank") == 4),
          ("rank 3~0 (relaxed)", lambda m: m.get("rank", 4) < 4),
          ("tier 3", lambda m: m.get("tier") == 3),
          ("AI=shipped tex", lambda m: m.get("src") == "shipped_tex"),
          ("AI=rebuilt from PDF", lambda m: m.get("src") == "rebuilt_from_pdf")]
cks = [c for c in sorted(os.listdir(f"{B}/results/slop"))
       if os.path.isfile(f"{B}/results/slop/{c}/papers.jsonl")]
print(f"{'split':<20} " + "  ".join(f"{c[:13]:>16}" for c in cks))
for name, keep in SPLITS:
    cells = []
    for c in cks:
        r = score(c, keep)
        cells.append("               -" if r is None else f"{r[0]:>6.3f}/{r[1]:<5.3f} n{r[2]:<3d}")
    print(f"{name:<20} " + "  ".join(cells))
print("\ncells are PairAcc/AUROC n=pairs")
