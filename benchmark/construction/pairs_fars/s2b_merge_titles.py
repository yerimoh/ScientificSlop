"""Step 2b: merge title-lookup resolutions (cache/title2id.json) back into the candidate pool.

Adds each resolved arXiv id to the pool of every FARS code that cited that title,
tagged source "bib_title_lookup". Updates cache/pool.json and cache/all_ids.json.
"""
import os
import re, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911"


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


pool = json.load(open(f"{OUTD}/cache/pool.json"))
queue = json.load(open(f"{OUTD}/cache/title_queue.json"))
t2i = json.load(open(f"{OUTD}/cache/title2id.json"))

added = 0
for q in queue:
    aid = t2i.get(norm_title(q["title"]))
    if not aid:
        continue
    aid = re.sub(r"v\d+$", "", aid)
    for code in q["codes"]:
        P = pool[code]
        if aid not in P:
            P[aid] = dict(sources=[], bib_keys=[], cand_meta=None)
            added += 1
        if "bib_title_lookup" not in P[aid]["sources"]:
            P[aid]["sources"].append("bib_title_lookup")
        for ck in q.get("keys", []):
            c, k = ck.split(":", 1)
            if c == code and k not in P[aid]["bib_keys"]:
                P[aid]["bib_keys"].append(k)

json.dump(pool, open(f"{OUTD}/cache/pool.json", "w"), ensure_ascii=False)
allids = sorted({a for P in pool.values() for a in P})
json.dump(allids, open(f"{OUTD}/cache/all_ids.json", "w"))
resolved = sum(1 for v in t2i.values() if v)
print(f"resolved titles {resolved}/{len(t2i)} | new (code,id) additions {added} | unique ids now {len(allids)}")
