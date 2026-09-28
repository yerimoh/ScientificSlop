"""Step 2 (Agents4Science): per-paper candidate pool from the parsed reference list.

A reference enters the pool of its citing paper once it has an arXiv id. Ids come from the
entry text itself (arXiv:XXXX.XXXXX) and, for the rest, from the title queue that a3 resolves.
Venue-name and tool junk is dropped before queueing, as in the FARS build.

Outputs: cache/pool.json        {code: {arxiv_id: {sources, ref_idx, cite_roles}}}
         cache/title_queue.json [{title, codes, refs}]
"""
import os, re, sys, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"

JUNK = re.compile(r"^(science|nature|arxiv|proceedings|conference|journal|ieee|acm|advances in)\b|"
                  r"(conference|symposium|workshop|journal|transactions) (on|of)\b", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main():
    idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
    t2i_path = f"{OUTD}/cache/title2id.json"
    t2i = json.load(open(t2i_path)) if os.path.exists(t2i_path) else {}
    # seed from the FARS build: same normalization, ~900 titles already resolved
    seed = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911/cache/title2id.json"
    n_seed = 0
    if os.path.exists(seed):
        for k, v in json.load(open(seed)).items():
            if k not in t2i:
                t2i[k] = v
                n_seed += 1
    json.dump(t2i, open(t2i_path, "w"), ensure_ascii=False)

    pool, tq = {}, {}
    for code, rec in idx.items():
        P = {}
        for r in rec["refs"]:
            roles = rec["cite_roles"].get(str(r["idx"]), {})
            aid = r["arxiv"]
            if not aid:
                nt = norm_title(r["title"])
                if nt and t2i.get(nt):
                    aid = t2i[nt]
            if aid:
                aid = re.sub(r"v\d+$", "", aid)
                P.setdefault(aid, dict(sources=[], ref_idx=[], cite_roles={}))
                P[aid]["sources"].append("bib_arxiv" if r["arxiv"] else "title_lookup")
                P[aid]["ref_idx"].append(r["idx"])
                for k, v in roles.items():
                    P[aid]["cite_roles"][k] = P[aid]["cite_roles"].get(k, 0) + v
            elif r["title"] and len(r["title"].split()) >= 4 and not JUNK.search(r["title"]):
                nt = norm_title(r["title"])
                if nt in t2i:      # already looked up and unresolved
                    continue
                tq.setdefault(nt, dict(title=r["title"], codes=set(), refs=set()))
                tq[nt]["codes"].add(code)
                tq[nt]["refs"].add(f"{code}:{r['idx']}")
        pool[code] = P
    for v in tq.values():
        v["codes"], v["refs"] = sorted(v["codes"]), sorted(v["refs"])

    json.dump(pool, open(f"{OUTD}/cache/pool.json", "w"), ensure_ascii=False)
    json.dump(list(tq.values()), open(f"{OUTD}/cache/title_queue.json", "w"), ensure_ascii=False, indent=1)
    allids = sorted({a for P in pool.values() for a in P})
    json.dump(allids, open(f"{OUTD}/cache/all_ids.json", "w"))
    ns = sorted(len(P) for P in pool.values())
    print(f"seeded {n_seed} titles from the FARS build | papers {len(pool)} | unique arXiv ids {len(allids)}")
    print(f"pool sizes min/med/max {ns[0]}/{ns[len(ns)//2]}/{ns[-1]} | zero-pool papers "
          f"{sum(1 for P in pool.values() if not P)} | title queue {len(tq)}")


if __name__ == "__main__":
    main()
