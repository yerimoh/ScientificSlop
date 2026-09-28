"""Fill the reviewer coverage of the multi-year corpus so every year can carry a per-year accept-against-reject
comparison.

The first pass drew a score-stratified sample, which left 7 to 29 rejected and 13 to 25 accepted papers a year,
below the 20 a side that the per-year comparison needs. This lists what is missing, taking papers at random inside
each year and decision cell until the cell holds TARGET, and never looking at any measured score.
Output: records/reviewer_subset_years_fill.json
"""
import csv, glob, json, os, random, collections

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = int(os.environ.get("TARGET", "45"))
random.seed(0)


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


done = scored("b2h") & scored("b3a")
rows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv")) if r["group"] in ("reject", "accept", "oral")]
cells = collections.defaultdict(list)
for r in rows:
    if r["id"] in done:
        continue
    cells[(r["year"], "reject" if r["group"] == "reject" else "accept")].append(r["id"])
have = collections.Counter()
for r in rows:
    if r["id"] in done:
        have[(r["year"], "reject" if r["group"] == "reject" else "accept")] += 1
ids, plan = [], {}
for (y, g), pool in sorted(cells.items()):
    need = max(0, TARGET - have[(y, g)])
    random.shuffle(pool)
    take = pool[:need]
    ids += take
    plan[f"{y} {g}"] = {"already": have[(y, g)], "adding": len(take), "pool": len(pool)}
json.dump({"rule": f"random fill to {TARGET} papers per year and decision, never looking at a measured score",
           "target": TARGET, "n": len(ids), "plan": plan, "ids": ids},
          open(f"{HERE}/records/reviewer_subset_years_fill.json", "w"), indent=1)
print(f"already reviewed {len(done)}, adding {len(ids)} to reach {TARGET} per year and decision")
for k, v in plan.items():
    print(f"  {k:16s} have {v['already']:3d}  add {v['adding']:3d}  (pool {v['pool']})")
