"""Step 2: per-FARS candidate pool = candidates.json (pre-filtered top-tier cited) + arXiv ids
scraped from the cited analemma.bib entries. Cited bib entries with no arXiv id anywhere go to a
global title-lookup queue (deduplicated; obvious venue-name junk filtered).

Outputs: cache/pool.json           {code: {arxiv_id: {sources, bib_keys, cand_meta}}}
         cache/title_queue.json    [{title, codes: [FA..], keys: [...]}]
"""
import os, re, sys, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911"

idx = json.load(open(f"{OUTD}/cache/fars_index.json"))
cands = json.load(open(f"{ROOT}/fars/analysis/05_anchor_pairs/candidates.json"))

JUNK = re.compile(r"^(science|nature|arxiv|proceedings|conference|journal|ieee|acm|advances in)\b|"
                  r"(conference|symposium|workshop|journal|transactions) (on|of)\b", re.I)

def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

pool = {}
tq = {}
for code, rec in idx.items():
    P = {}
    # source A: candidates.json (pre-filtered top-tier citations)
    for c in cands.get(code, {}).get("candidates", []):
        aid = c.get("arxiv_id")
        entry = dict(ref_title=c["ref_title"], venue=c.get("venue"), year=c.get("year"),
                     citations=c.get("citations"), s2_id=c.get("s2_id"))
        if aid:
            aid = re.sub(r"v\d+$", "", aid)
            P.setdefault(aid, dict(sources=[], bib_keys=[], cand_meta=None))
            P[aid]["sources"].append("candidates.json")
            P[aid]["cand_meta"] = entry
        else:
            t = c["ref_title"]
            if len(t.split()) >= 4 and not JUNK.search(t):
                tq.setdefault(norm_title(t), dict(title=t, codes=set(), keys=set()))
                tq[norm_title(t)]["codes"].add(code)
    # source B: cited bib entries
    for key in rec["cite_roles"]:
        b = rec["bib"].get(key)
        if not b:
            continue
        if b["arxiv"]:
            aid = re.sub(r"v\d+$", "", b["arxiv"])
            P.setdefault(aid, dict(sources=[], bib_keys=[], cand_meta=None))
            P[aid]["sources"].append("bib")
            P[aid]["bib_keys"].append(key)
        elif b["title"] and len(b["title"].split()) >= 4 and not JUNK.search(b["title"]):
            nt = norm_title(b["title"])
            tq.setdefault(nt, dict(title=b["title"], codes=set(), keys=set()))
            tq[nt]["codes"].add(code)
            tq[nt]["keys"].add(f"{code}:{key}")
    pool[code] = P

for v in tq.values():
    v["codes"] = sorted(v["codes"])
    v["keys"] = sorted(v["keys"])

json.dump(pool, open(f"{OUTD}/cache/pool.json", "w"), ensure_ascii=False)
json.dump(list(tq.values()), open(f"{OUTD}/cache/title_queue.json", "w"), ensure_ascii=False, indent=1)
allids = sorted({a for P in pool.values() for a in P})
json.dump(allids, open(f"{OUTD}/cache/all_ids.json", "w"))
ns = sorted(len(P) for P in pool.values())
print("FARS:", len(pool), "| unique arxiv ids:", len(allids), "| title queue:", len(tq))
print("pool sizes min/med/max:", ns[0], ns[len(ns)//2], ns[-1])
print("zero-pool codes:", [c for c, P in pool.items() if not P])
