"""Step 10 (Agents4Science): give every remaining paper an anchor, so the benchmark carries all
247 pairs.

Steps 6 and 9 assign only where a candidate is both the same contribution type and at least a
topic neighbour (tier 2), which the ICLR 2026 pool cannot supply for the parts of the corpus that
are physics, biology, medicine or finance. The conditions that decide whether an anchor is a valid
human paper at all stay hard here, H1 Pangram-verified human, H2 accepted, H4 an e-print whose
LaTeX expands to a body of at least 2,500 words in at least four sections. What is relaxed is the
match, in this order:

    tier >= 2 and the same type        (what steps 6 and 9 already took)
    tier >= 2, different type
    tier 1, same type
    tier 1, different type
    the nearest unjudged neighbour by TF-IDF

Every pair records sim_tier, type_match and the caveats the relaxation earns, so an analysis can
keep only the matched pairs or use all 247 and say which is which.

Outputs: pairs_a4s.json (in place), cache/a10_fill_report.json
"""
import json, os, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"
sys.path.insert(0, f"{DATA}/scripts")
from slopbench_lib import flatten_dir, metrics  # noqa: E402
import io, gzip, tarfile, time, urllib.request


def eprint(ax, dest):
    """Same downloader as step 9, copied rather than imported because that module assigns on import."""
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 1:
        return True
    os.makedirs(dest, exist_ok=True)
    try:
        req = urllib.request.Request(f"https://export.arxiv.org/e-print/{ax}",
                                     headers={"User-Agent": "scislopbench-a4s/0.1 (user@example.org)"})
        b = urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:
        print("  download fail", ax, e)
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(b)) as tf:
            tf.extractall(dest, members=[m for m in tf.getmembers()
                                         if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in m.name])
    except Exception:
        try:
            open(os.path.join(dest, "main.tex"), "wb").write(gzip.decompress(b))
        except Exception as e:
            print("  extract fail", ax, e)
            return False
    time.sleep(3)
    return True

short = json.load(open(f"{OUTD}/cache/widen_shortlist_full.json"))
judge = json.load(open(f"{OUTD}/cache/widen_judge.json"))
types = json.load(open(f"{OUTD}/cache/a4s_types.json"))
idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
draft = json.load(open(f"{OUTD}/pairs_a4s.json"))
taken = {p["human_primary"]["arxiv"] for p in draft["pairs"] if p.get("human_primary")}
open_codes = [p["code"] for p in draft["pairs"] if not p.get("human_primary")]
print("to fill:", len(open_codes))


def rank(code):
    """Candidates worst-last, by how well they match, then ICLR rating, then TF-IDF."""
    ft, js = types[code], judge.get(code, {}).get("cands", {})
    out = []
    for c in short.get(code, []):
        ax = c["arxiv"]
        if ax in taken:
            continue
        j = js.get(ax)
        tier = j["tier"] if j else None
        same = (j["type"] == ft) if j else None
        if tier is not None and tier >= 2 and same:
            k = 4
        elif tier is not None and tier >= 2:
            k = 3
        elif tier == 1 and same:
            k = 2
        elif tier == 1:
            k = 1
        else:
            k = 0
        out.append((k, tier if tier is not None else -1, float(c.get("rating") or 0), c["sim"], c, j))
    out.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3]))
    return out


def h4(ax):
    dest = f"{OUTD}/eprints_new/{ax}"
    alt = f"{ROOT}/ana/reference/data/human_refs/eprints/{ax}"
    src = alt if os.path.isdir(alt) else dest
    if src == dest and not eprint(ax, dest):
        return None, "no e-print"
    r, full = flatten_dir(src)
    if r is None:
        return None, "no root tex"
    m = metrics(full)
    if m["body_words"] < 2500 or m["n_sections"] < 4:
        return None, f"body {m['body_words']} sections {m['n_sections']}"
    return (src, r, m, "human_refs/eprints/" if src == alt else "a4s/eprints_new/"), None


CAVEAT = {4: [], 3: ["contribution_type_mismatch"], 2: ["similarity_tier_1"],
          1: ["similarity_tier_1", "contribution_type_mismatch"],
          0: ["similarity_tier_0_or_unjudged", "contribution_type_unchecked"]}
final, failed = {}, {}
for code in open_codes:
    got = None
    for k, tier, rating, sim, c, j in rank(code):
        ax = c["arxiv"]
        src, why = h4(ax)
        if src is None:
            failed.setdefault(code, []).append(f"{ax}: {why}")
            continue
        taken.add(ax)
        got = (k, tier, sim, c, j, src)
        break
    if got is None:
        print(code, "NO ANCHOR", failed.get(code))
        continue
    k, tier, sim, c, j, (srcd, r, m, srclabel) = got
    final[code] = dict(arxiv=c["arxiv"], title=c["title"], v1_year=None,
                       venue=f"ICLR 2026 {c['tier']}", paper_type=(j or {}).get("type"),
                       sim_tier=tier if tier >= 0 else None,
                       tier_reason=(j or {}).get("reason", "nearest topic neighbour, not judged"),
                       relation="filled (nearest admissible anchor)", cite_roles="{}",
                       H1_human="pass", H1_by="pangram", H2_top_tier="pass",
                       H3_type_match="pass" if k in (4, 2) else "relaxed", H4_source="pass",
                       pangram_ai_frac=c["fai"], iclr_rating=c["rating"], iclr_tier=c["tier"],
                       tfidf_sim=sim, body_words=m["body_words"], body_ratio=None,
                       appendix_words=m["appendix_words"], n_sections=m["n_sections"],
                       n_figures=m["n_figures"], n_tables=m["n_tables"],
                       n_unique_cites=m["n_unique_cites"], n_internal_refs=m["n_internal_refs"],
                       has_method_fig=None, github=";".join(m["github"][:2]),
                       template=";".join(m["template"][:1]), tex_root=os.path.relpath(r, srcd),
                       tex_source=srclabel + c["arxiv"], match_rank=k)
    print(code, "->", c["arxiv"], f"rank {k} tier {tier} sim {sim} body {m['body_words']}")

for p in draft["pairs"]:
    hp = final.get(p["code"])
    if not hp:
        continue
    bw = idx[p["code"]]["metrics"]["body_words"]
    hp["body_ratio"] = round(hp["body_words"] / bw, 2) if bw else None
    p["human_primary"] = hp
    p["pool"] = "filled_iclr26"
    p["no_candidate"] = False
    p["no_candidate_reason"] = None
    p["caveats"] = sorted(set(p.get("caveats", []) + CAVEAT[hp["match_rank"]]))
draft["n_primary"] = sum(1 for p in draft["pairs"] if p.get("human_primary"))
draft["n_no_candidate"] = len(draft["pairs"]) - draft["n_primary"]
draft["filled"] = sorted(final)
json.dump(draft, open(f"{OUTD}/pairs_a4s.json", "w"), indent=1, ensure_ascii=False)
json.dump({"filled": {k: v["match_rank"] for k, v in final.items()}, "h4_failures": failed},
          open(f"{OUTD}/cache/a10_fill_report.json", "w"), indent=1)
print(f"\nTOTAL primary {draft['n_primary']} / {len(draft['pairs'])}  (filled {len(final)})")
