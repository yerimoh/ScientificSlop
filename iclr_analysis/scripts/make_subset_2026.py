"""Reviewer subset for ICLR 2026. The two reviewer models take about three minutes a paper, so the whole corpus is
out of reach. We draw an equal number per review-score bin (rounded to the half point, bins with at least 12 papers),
which keeps the score range that the analysis needs. Selection never looks at any measured score.
Output: records/reviewer_subset_2026.json
"""
import json, os, collections, random
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER = int(os.environ.get("PER", "40"))
random.seed(0)
P = [json.loads(l) for l in open(f"{HERE}/results/papers_2026.jsonl")]
bins = collections.defaultdict(list)
for v in P:
    bins[round(v["rating"] * 2) / 2].append(v["id"])
ids = []
for b in sorted(bins):
    pool = sorted(bins[b])
    if len(pool) < 12:
        continue
    random.shuffle(pool)
    ids += pool[:PER]
json.dump({"rule": f"{PER} papers per half-point review-score bin with at least 12 papers, random, seed 0",
           "bins": {str(b): len(v) for b, v in sorted(bins.items())}, "n": len(ids), "ids": ids},
          open(f"{HERE}/records/reviewer_subset_2026.json", "w"), indent=1)
print("subset", len(ids), "from bins", {str(b): len(v) for b, v in sorted(bins.items()) if len(v) >= 12})
