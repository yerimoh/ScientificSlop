"""Step 8: widening (PAIR_RULE_0911 step 1, last clause) for FARS papers whose CITED pool
left no assignable candidate. Pool = ICLR 2026 submissions that are Pangram-verified human
(fraction_ai <= 0.05), ACCEPTED (H2), and carry an arXiv id (so H4 can be checked later).
Shortlist = top-K TF-IDF neighbours of the FARS title+abstract. Types/tiers are judged
afterwards (cache/widen_judge.json), then s9_widen_assign.py merges them in.
"""
import json, os, sys
ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911"
K = 3

d = json.load(open(f"{OUTD}/pairs165_draft.json"))
nc = [p for p in d["pairs"] if not p.get("human_primary")]
used = {p["human_primary"]["arxiv"] for p in d["pairs"] if p.get("human_primary")}
idx = json.load(open(f"{OUTD}/cache/fars_index.json"))

pool = []
seen = set()
with open(f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        ax = (r.get("arxiv_id") or "").split("v")[0]
        fai = r.get("fraction_ai")
        if not ax or ax == "None" or ax in used or ax in seen:
            continue
        if fai in (None, "None") or float(fai) > 0.05:
            continue
        if not r.get("accept") in (True, "True", "true"):
            continue
        seen.add(ax)
        pool.append(dict(arxiv=ax, sub=r["submission_id"], title=r["title"], abstract=r.get("abstract") or "",
                         rating=r.get("rating_mean"), tier=r.get("tier"), fai=float(fai)))
print("widen pool (human-verified accepted ICLR26 w/ arXiv id):", len(pool))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = [p["title"] + ". " + p["abstract"] for p in pool]
qs = {p["code"]: (idx[p["code"]]["title"] + ". " + (idx[p["code"]].get("abstract") or "")) for p in nc}
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(docs + list(qs.values()))
X = vec.transform(docs)

short = {}
taken = set()
for code, q in qs.items():
    s = cosine_similarity(vec.transform([q]), X)[0]
    order = s.argsort()[::-1]
    picks = []
    for j in order:
        if pool[j]["arxiv"] in taken:      # soft de-dup across shortlists so top-1s spread out
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
        w.write(f"\n===== {code} [{p.get('fars_type')}] {f['title']}\n")
        w.write((f.get("abstract") or "")[:260] + "\n")
        for c in short[code]:
            w.write(f"  -- {c['arxiv']} (r={c['rating']}, {c['tier']}, sim={c['sim']}) {c['title']}\n")
            w.write("     " + c["abstract"][:280].replace("\n", " ") + "\n")
print("shortlists:", len(short), "-> cache/widen_shortlist.json, cache/widen_listing.txt")
