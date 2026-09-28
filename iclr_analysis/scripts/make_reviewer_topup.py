"""Top up the reviewer subset so every review score 2..9 has >= PER clean papers with reviews; new ids only."""
import json, collections, random, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 30
random.seed(1)
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v["group"] in ("reject", "accept", "oral") and v.get("rating") is not None}
done = {os.path.basename(f)[:-5] for f in os.listdir(f"{HERE}/results/reviews/b2h")} | {os.path.basename(f)[:-5] for f in os.listdir(f"{HERE}/results/reviews/b3a")}
have = collections.Counter(int(round(iclr[i]["rating"])) for i in done if i in iclr)
top = []
for s in range(2, 10):
    need = PER - have[s]
    cand = [i for i, v in iclr.items() if int(round(v["rating"])) == s and i not in done]
    random.shuffle(cand); top += cand[:max(0, need)]
json.dump({"rule": f"{PER} per integer score 2..9, random, clean papers, excluding already reviewed", "ids": top}, open(f"{HERE}/records/reviewer_subset_topup.json", "w"), indent=1)
print("already reviewed per score:", dict(sorted(have.items())), "| top-up:", len(top))
