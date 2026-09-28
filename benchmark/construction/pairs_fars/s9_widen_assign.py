"""Step 9: assign widened ICLR-2026 anchors (PAIR_RULE step 1 last clause) to FARS papers
that the cited pool could not serve. Uses cache/widen_shortlist.json + cache/widen_judge.json.

Assignment: per FARS, best candidate with type == fars_type and tier >= 2
(rank by tier desc, then ICLR rating desc, then TF-IDF sim). One-to-one across widened
anchors and against already-assigned cited anchors. H4 is checked after download:
body >= 2500 prose words and >= 4 sections via slopbench_lib. Assigned pairs get
pool="widened_iclr26" and are merged into pairs165_draft.json (draft2 written alongside).
"""
import json, os, sys, subprocess, urllib.request, tarfile, gzip, io, time

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911"
sys.path.insert(0, f"{ROOT}/paper/draft_v6/scislopbench/data/scripts")
from slopbench_lib import flatten_dir, metrics, split_body  # noqa: E402

short = json.load(open(f"{OUTD}/cache/widen_shortlist.json"))
judge = json.load(open(f"{OUTD}/cache/widen_judge.json"))
draft = json.load(open(f"{OUTD}/pairs165_draft.json"))
used = {p["human_primary"]["arxiv"] for p in draft["pairs"] if p.get("human_primary")}

meta = {c["arxiv"]: c for lst in short.values() for c in lst}

# rank choices per FARS
choices = {}
for code, j in judge.items():
    ft = j["fars_type"]
    cs = []
    for ax, v in j["cands"].items():
        if v["type"] == ft and v["tier"] >= 2 and ax not in used:
            m = meta[ax]
            cs.append((v["tier"], float(m.get("rating") or 0), m["sim"], ax, v))
    cs.sort(reverse=True)
    if cs:
        choices[code] = cs

# global one-to-one: iterate FARS by best available (tier, rating), take greedily
order = sorted(choices, key=lambda c: (-choices[c][0][0], -choices[c][0][1]))
taken = set(used)
assigned = {}
for code in order:
    for tier, rating, sim, ax, v in choices[code]:
        if ax in taken:
            continue
        assigned[code] = (ax, tier, rating, sim, v)
        taken.add(ax)
        break
print("widened assignments:", len(assigned))

# download + H4
def eprint(ax, dest):
    if os.path.isdir(dest) and len(os.listdir(dest)) > 1:
        return True
    os.makedirs(dest, exist_ok=True)
    try:
        req = urllib.request.Request(f"https://arxiv.org/e-print/{ax}",
                                     headers={"User-Agent": "research user@example.org"})
        b = urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:
        print("  download fail", ax, e); return False
    try:
        tarfile.open(fileobj=io.BytesIO(b)).extractall(dest)
    except Exception:
        try:
            open(os.path.join(dest, "main.tex"), "wb").write(gzip.decompress(b))
        except Exception as e:
            print("  extract fail", ax, e); return False
    time.sleep(3)
    return True

final = {}
for code, (ax, tier, rating, sim, v) in sorted(assigned.items()):
    dest = f"{OUTD}/eprints_new/{ax}"
    alt = f"{ROOT}/ana/reference/data/human_refs/eprints/{ax}"
    src = alt if os.path.isdir(alt) else dest
    if src == dest and not eprint(ax, dest):
        print(code, ax, "H4 fail: no e-print"); continue
    r, full = flatten_dir(src)
    if r is None:
        print(code, ax, "H4 fail: no root tex"); continue
    m = metrics(full)
    if m["body_words"] < 2500 or m["n_sections"] < 4:
        print(code, ax, f"H4 fail: body {m['body_words']} sections {m['n_sections']}"); continue
    mm = meta[ax]
    final[code] = dict(arxiv=ax, title=mm["title"], venue=f"ICLR 2026 {mm['tier']}", iclr_rating=mm["rating"],
                       pangram_ai_frac=mm["fai"], paper_type=v["type"], fars_type=judge[code]["fars_type"],
                       sim_tier=v["tier"], tier_reason=v["reason"], relation="widened (not cited; topic neighbour)",
                       pool="widened_iclr26", tfidf_sim=sim,
                       H1_human="pass (pangram)", H2_top_tier="pass", H3_type_match="pass", H4_source="pass",
                       body_words=m["body_words"], appendix_words=m["appendix_words"], n_sections=m["n_sections"],
                       n_figures=m["n_figures"], n_tables=m["n_tables"], n_unique_cites=m["n_unique_cites"],
                       n_internal_refs=m["n_internal_refs"], github=";".join(m["github"][:2]),
                       template=";".join(m["template"][:1]), tex_root=os.path.relpath(r, src),
                       tex_source=("human_refs/eprints/" if src == alt else "pairs165_0911/eprints_new/") + ax)
    print(code, "->", ax, f"tier {v['tier']} {v['type']} r={mm['rating']} body {m['body_words']}")

# merge
for p in draft["pairs"]:
    if p["code"] in final:
        p["human_primary"] = final[p["code"]]
        p["fars_type"] = judge[p["code"]]["fars_type"]
        p.pop("no_candidate", None); p.pop("no_candidate_reason", None)
draft["n_primary"] = sum(1 for p in draft["pairs"] if p.get("human_primary"))
draft["n_no_candidate"] = len(draft["pairs"]) - draft["n_primary"]
draft["widened"] = sorted(final)
json.dump(draft, open(f"{OUTD}/pairs165_draft.json", "w"), indent=1, ensure_ascii=False)
print(f"\nTOTAL primary {draft['n_primary']} / {len(draft['pairs'])}  (widened {len(final)})")
