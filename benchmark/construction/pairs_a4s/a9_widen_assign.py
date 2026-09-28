"""Step 9 (Agents4Science): assign widened ICLR-2026 anchors to papers the cited pool could not
serve. Needs cache/widen_shortlist.json + cache/widen_judge.json (type and tier per candidate).

Per paper the best candidate with type == a4s_type and tier >= 2 is taken, ranked by tier, then
ICLR rating, then TF-IDF similarity, one-to-one across the corpus and against anchors already
assigned from cited pools. H4 is checked after download. Merged into pairs_a4s.json with
pool="widened_iclr26".
"""
import json, os, sys, io, time, gzip, tarfile, urllib.request

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"
sys.path.insert(0, f"{DATA}/scripts")
from slopbench_lib import flatten_dir, metrics  # noqa: E402

short = json.load(open(f"{OUTD}/cache/widen_shortlist.json"))
judge = json.load(open(f"{OUTD}/cache/widen_judge.json"))
types = json.load(open(f"{OUTD}/cache/a4s_types.json"))
draft = json.load(open(f"{OUTD}/pairs_a4s.json"))
used = {p["human_primary"]["arxiv"] for p in draft["pairs"] if p.get("human_primary")}
meta = {c["arxiv"]: c for lst in short.values() for c in lst}

# only papers that the cited pool could not serve; rerunning must not touch existing primaries
open_codes = {p["code"] for p in draft["pairs"] if not p.get("human_primary")}
choices = {}
for code, j in judge.items():
    if code not in open_codes:
        continue
    ft = types[code]
    cs = []
    for ax, v in j["cands"].items():
        if v["type"] == ft and v["tier"] >= 2 and ax not in used and ax in meta:
            m = meta[ax]
            cs.append((v["tier"], float(m.get("rating") or 0), m["sim"], ax, v))
    cs.sort(reverse=True)
    if cs:
        choices[code] = cs

order = sorted(choices, key=lambda c: (-choices[c][0][0], -choices[c][0][1]))
taken, assigned = set(used), {}
for code in order:
    for tier, rating, sim, ax, v in choices[code]:
        if ax in taken:
            continue
        assigned[code] = (ax, tier, rating, sim, v)
        taken.add(ax)
        break
print("widened assignments proposed:", len(assigned))


def eprint(ax, dest):
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


def h4(ax):
    """Download if needed and return (src, root, metrics) when the e-print clears H4."""
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


final = {}
for code, (ax, tier, rating, sim, v) in sorted(assigned.items()):
    got, why = h4(ax)
    # H4 is only known after the download, so fall through to the next eligible candidate
    while got is None:
        print(code, ax, "H4 fail:", why)
        nxt = None
        for t2, r2, s2, ax2, v2 in choices[code]:
            if ax2 not in taken:
                nxt = (ax2, t2, r2, s2, v2)
                break
        if nxt is None:
            break
        ax, tier, rating, sim, v = nxt
        taken.add(ax)
        got, why = h4(ax)
    if got is None:
        continue
    src, r, m, srclabel = got
    mm = meta[ax]
    final[code] = dict(arxiv=ax, title=mm["title"], v1_year=None, venue=f"ICLR 2026 {mm['tier']}",
                       paper_type=v["type"], sim_tier=v["tier"], tier_reason=v["reason"],
                       relation="widened (not cited; topic neighbour)", cite_roles="{}",
                       H1_human="pass", H1_by="pangram", H2_top_tier="pass", H3_type_match="pass",
                       H4_source="pass", pangram_ai_frac=mm["fai"], iclr_rating=mm["rating"],
                       iclr_tier=mm["tier"], tfidf_sim=sim,
                       body_words=m["body_words"], body_ratio=None, appendix_words=m["appendix_words"],
                       n_sections=m["n_sections"], n_figures=m["n_figures"], n_tables=m["n_tables"],
                       n_unique_cites=m["n_unique_cites"], n_internal_refs=m["n_internal_refs"],
                       has_method_fig=None, github=";".join(m["github"][:2]),
                       template=";".join(m["template"][:1]), tex_root=os.path.relpath(r, src),
                       tex_source=srclabel + ax)
    print(code, "->", ax, f"tier {v['tier']} {v['type']} r={mm['rating']} body {m['body_words']}")

idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
for p in draft["pairs"]:
    if p["code"] in final:
        hp = final[p["code"]]
        bw = idx[p["code"]]["metrics"]["body_words"]
        hp["body_ratio"] = round(hp["body_words"] / bw, 2) if bw else None
        p["human_primary"] = hp
        p["pool"] = "widened_iclr26"
        p["no_candidate"] = False
        p["no_candidate_reason"] = None
draft["n_primary"] = sum(1 for p in draft["pairs"] if p.get("human_primary"))
draft["n_no_candidate"] = len(draft["pairs"]) - draft["n_primary"]
draft["widened"] = sorted(final)
json.dump(draft, open(f"{OUTD}/pairs_a4s.json", "w"), indent=1, ensure_ascii=False)
print(f"\nTOTAL primary {draft['n_primary']} / {len(draft['pairs'])}  (widened {len(final)})")
