"""Everything in the ICLR 2026 analysis set that no reviewer has scored yet.

The first reviewer pass took 40 papers per half-point review-score bin, which left the x axis of the single-year
figure with three usable points. To put every review score on that axis the reviewers have to cover the same papers
the detectors do, so this lists the rest of the analysis set and the runners walk it in shards.
Output: records/reviewer_subset_2026_rest.json
"""
import glob, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ana = [json.loads(l)["id"] for l in open(f"{HERE}/results/papers_2026.jsonl")]


def scored(sub):
    out = set()
    for f in glob.glob(f"{HERE}/results/reviews/{sub}/*.json"):
        try:
            r = json.load(open(f)); fin = r.get("final") or {}
        except Exception:
            continue
        if isinstance(fin, dict) and isinstance(fin.get("Overall", fin.get("Rating")), (int, float)):
            out.add(os.path.splitext(os.path.basename(f))[0])
    return out


done = scored("b2h_2026") & scored("b3a_2026")
rest = [p for p in ana if p not in done]
json.dump({"rule": "every paper of the 2026 analysis set that both reviewers have not scored yet", "n": len(rest),
           "ids": rest}, open(f"{HERE}/records/reviewer_subset_2026_rest.json", "w"), indent=1)
print(f"analysis set {len(ana)}, both reviewers already scored {len(done)}, still to do {len(rest)}")
