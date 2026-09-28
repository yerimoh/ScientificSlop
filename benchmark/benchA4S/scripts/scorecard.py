"""PairAcc and AUROC for every benchA4S item that has a result, from papers.jsonl.

Same reading as bench165: the score is the item's slop_score, one paper at a time, AI is the
positive class. PairAcc counts a pair only when both of its papers carry a score, and a tie is
half a win. Items that leave a paper NA (evidence_gap when a paper shows no result table) lose
that pair from the denominator rather than scoring it zero, so n is printed next to every number.
"""
import json, os, sys, statistics

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/itemsA4S.json"))["items"]
pair_of = {}
for i in items:
    key = i["pair"] if i["label"] == 1 else i.get("arxiv")
    pair_of[("AI" if i["label"] == 1 else "HU", key)] = i["pair"]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    n = 0.0
    for a in pos:
        for b in neg:
            n += 1.0 if a > b else 0.5 if a == b else 0.0
    return round(n / (len(pos) * len(neg)), 4)


def score_of(r, field):
    v = r.get(field)
    return None if v is None else float(v)


rows = []
for ck in sorted(os.listdir(f"{B}/results/slop")):
    f = f"{B}/results/slop/{ck}/papers.jsonl"
    if not os.path.isfile(f):
        continue
    recs = [json.loads(l) for l in open(f)]
    field = "slop_score" if any("slop_score" in r for r in recs) else None
    if field is None:
        rows.append((ck, len(recs), None, None, None, "no slop_score"))
        continue
    by = {}
    for r in recs:
        p = pair_of.get((r["corpus"], r["id"]))
        if p is None:
            continue
        by.setdefault(p, {})[r["corpus"]] = score_of(r, field)
    ai = [v["AI"] for v in by.values() if v.get("AI") is not None]
    hu = [v["HU"] for v in by.values() if v.get("HU") is not None]
    both = [(v["AI"], v["HU"]) for v in by.values()
            if v.get("AI") is not None and v.get("HU") is not None]
    wins = sum(1.0 if a > h else 0.5 if a == h else 0.0 for a, h in both)
    pa = round(wins / len(both), 4) if both else None
    wlt = (sum(1 for a, h in both if a > h), sum(1 for a, h in both if a < h),
           sum(1 for a, h in both if a == h))
    rows.append((ck, len(by), pa, auroc(ai, hu), (len(both), len(ai), len(hu)), wlt))

w = max(len(r[0]) for r in rows)
print(f"{'item':<{w}}  {'PairAcc':>8}  {'AUROC':>7}  {'pairs':>5}  {'AI':>4} {'HU':>4}   W/L/T")
for ck, npair, pa, au, ns, wlt in rows:
    if pa is None and au is None:
        print(f"{ck:<{w}}  {'-':>8}  {'-':>7}   {wlt}")
        continue
    nb, na, nh = ns
    print(f"{ck:<{w}}  {pa if pa is not None else '-':>8}  {au if au is not None else '-':>7}  "
          f"{nb:>5}  {na:>4} {nh:>4}   {wlt[0]}/{wlt[1]}/{wlt[2]}")
