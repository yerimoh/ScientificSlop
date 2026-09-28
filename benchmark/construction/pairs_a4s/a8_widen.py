"""Step 8 (Agents4Science): widening (PAIR_RULE_0911 step 1, last clause) for papers whose
CITED pool left no assignable anchor. Pool = ICLR 2026 submissions that are Pangram-verified
human (fraction_ai <= 0.05), accepted, and carry an arXiv id. Shortlist = top-K TF-IDF
neighbours of the A4S title+abstract; types and tiers are judged afterwards.

Outputs: cache/widen_shortlist.json, cache/widen_listing.txt
"""
import json, os, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"
K = int(os.environ.get("WIDEN_K", 3))

d = json.load(open(f"{OUTD}/pairs_a4s.json"))
nc = [p for p in d["pairs"] if not p.get("human_primary")]
used = {p["human_primary"]["arxiv"] for p in d["pairs"] if p.get("human_primary")}
idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
types = json.load(open(f"{OUTD}/cache/a4s_types.json"))

pool, seen = [], set()
with open(f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        ax = (r.get("arxiv_id") or "").split("v")[0]
        fai = r.get("fraction_ai")
        if not ax or ax == "None" or ax in used or ax in seen:
            continue
        if fai in (None, "None") or float(fai) > 0.05:
            continue
        if r.get("accept") not in (True, "True", "true"):
            continue
        seen.add(ax)
        pool.append(dict(arxiv=ax, sub=r["submission_id"], title=r["title"], abstract=r.get("abstract") or "",
                         rating=r.get("rating_mean"), tier=r.get("tier"), fai=float(fai)))
print("widen pool (human-verified accepted ICLR26 with an arXiv id):", len(pool))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = [p["title"] + ". " + p["abstract"] for p in pool]
qs = {p["code"]: (idx[p["code"]]["title"] + ". " + (idx[p["code"]].get("abstract") or "")) for p in nc}
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(docs + list(qs.values()))
X = vec.transform(docs)

short, taken = {}, set()
for code, q in qs.items():
    s = cosine_similarity(vec.transform([q]), X)[0]
    picks = []
    for j in s.argsort()[::-1]:
        if pool[j]["arxiv"] in taken:
            continue
        picks.append(dict(pool[j], sim=round(float(s[j]), 3)))
        if len(picks) == K:
            break
    for p_ in picks[:1]:
        taken.add(p_["arxiv"])
    short[code] = picks
json.dump(short, open(f"{OUTD}/cache/widen_shortlist.json", "w"), indent=1, ensure_ascii=False)

with open(f"{OUTD}/cache/widen_listing.txt", "w") as w:
    for p in nc:
        code = p["code"]
        f = idx[code]
        w.write(f"\n===== {code} [{types[code]}] {f['title']}\n")
        w.write(" ".join((f.get("abstract") or "")[:300].split()) + "\n")
        for c in short[code]:
            w.write(f"  -- {c['arxiv']} (r={c['rating']}, {c['tier']}, sim={c['sim']}) {c['title']}\n")
            w.write("     " + " ".join(c["abstract"][:300].split()) + "\n")
print("shortlists:", len(short), "-> cache/widen_shortlist.json, cache/widen_listing.txt")
