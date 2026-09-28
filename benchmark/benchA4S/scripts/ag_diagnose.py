"""Which Argument_Graph number separates the two corpora, and which does not.

The scorecard reads slop_score, the declared share of key claims, because that is the item's
score. This prints every numeric field the checker records, both directions, so a weak score can
be told apart from a weak item. Run on benchA4S and on bench165 to see whether a field is weak
here only or weak on both benches.
"""
import json, os, sys, statistics as st

BENCH = sys.argv[1] if len(sys.argv) > 1 else "benchA4S"
SB = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench"
items = json.load(open(f"{SB}/{BENCH}/items{'A4S' if BENCH=='benchA4S' else '165'}.json"))["items"]
key_of = {}
for i in items:
    k = i["pair"] if i["label"] == 1 else i.get("arxiv", i["item_id"].replace("HU_", ""))
    key_of[("AI" if i["label"] == 1 else "HU", k)] = i["pair"]

rows = [json.loads(l) for l in open(f"{SB}/{BENCH}/results/slop/argument_graph/papers.jsonl")]
flat = []
for r in rows:
    p = key_of.get((r["corpus"], r["id"]))
    if p is None:
        continue
    d = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k, v in (r.get("key_by_kind") or {}).items():
        d[f"kind_{k}"] = v
    for k, v in (r.get("slop_by_kind") or {}).items():
        if v is not None:
            d[f"slopkind_{k}"] = v
    flat.append((p, r["corpus"], d))

fields = sorted({k for _, _, d in flat for k in d})
by = {}
for p, c, d in flat:
    by.setdefault(p, {})[c] = d


def score(field, d):
    both = [(v["AI"][field], v["HU"][field]) for v in by.values()
            if "AI" in v and "HU" in v and field in v["AI"] and field in v["HU"]]
    if len(both) < 5:
        return None
    w = sum(1 for a, h in both if (a - h) * d > 0)
    t = sum(1 for a, h in both if a == h)
    ai = [v["AI"][field] for v in by.values() if "AI" in v and field in v["AI"]]
    hu = [v["HU"][field] for v in by.values() if "HU" in v and field in v["HU"]]
    au = sum(1.0 if (a - b) * d > 0 else 0.5 if a == b else 0.0
             for a in ai for b in hu) / (len(ai) * len(hu))
    return (w + 0.5 * t) / len(both), au, len(both), st.mean(ai), st.mean(hu)


out = []
for f in fields:
    for d in (+1, -1):
        r = score(f, d)
        if r:
            out.append((r[0], r[1], f, d, r[2], r[3], r[4]))
out = [o for o in out if o[0] >= 0.5]
out.sort(reverse=True)
print(f"{BENCH}  AG per-field separability (PairAcc descending)")
print(f"{'field':<28}{'dir':>7}{'PairAcc':>9}{'AUROC':>8}{'pairs':>5}   {'AI mean':>9} {'HU mean':>9}")
for pa, au, f, d, n, ma, mh in out[:24]:
    print(f"{f:<28}{('AI high' if d>0 else 'AI low'):>7}{pa:>8.3f}{au:>8.3f}{n:>5}   {ma:>9.3f} {mh:>9.3f}")
