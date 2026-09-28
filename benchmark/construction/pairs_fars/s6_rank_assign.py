"""Step 6: final H3 with judged types, ranking (PAIR_RULE_0911 step 3), global one-to-one
assignment, and outputs.

The five 0911 v3 pairs are frozen and copied through from pairs5_0911_v3.json.
Candidates with H4=unknown that land in the top 3 of a FARS are written to
cache/download_queue.json; run s7_download.py + s4_cand_metrics.py, then rerun this script.

Outputs: ../candidates_165.csv, ../pairs165_draft.json, cache/assign_report.json
"""
import os, re, csv, json, glob, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/pairs165_0911"
RATIO_MAX = 2.5

idx = json.load(open(f"{OUTD}/cache/fars_index.json"))
filters = json.load(open(f"{OUTD}/cache/filters.json"))
shortlists = json.load(open(f"{OUTD}/cache/shortlists.json"))
cm = json.load(open(f"{OUTD}/cache/cand_metrics_165.json"))
meta = json.load(open(f"{OUTD}/cache/arxiv_meta_165.json"))
pairs5 = json.load(open(f"{DATA}/pairs5_0911_v3.json"))

FIXED = {p["code"]: p for p in pairs5["pairs"]}  # FA0001/0209/0035/0046/0006

judged = {}
for f in sorted(glob.glob(f"{OUTD}/cache/judge_out/batch_*.json")):
    for rec in json.load(open(f)):
        judged[rec["code"]] = rec

rows_out = []
per_fars = {}
for code, srows in filters.items():
    if code in FIXED:
        continue
    j = judged.get(code)
    ftype = j["fars_type"] if j else shortlists[code]["fars_heur_type"]
    cands = []
    for aid, r in srows.items():
        if r.get("error"):
            rows_out.append(dict(code=code, arxiv=aid, note=r["error"]))
            continue
        jc = (j or {}).get("candidates", {}).get(aid)
        ctype = jc["type"] if jc else r["heur_type"]
        tier = jc["tier"] if jc else None
        reason = jc["reason"] if jc else ""
        type_src = "judged" if jc else "heuristic"
        h3 = "pass" if ctype == ftype else "fail"
        c = cm.get(aid)
        mt = c["metrics"] if c and c.get("metrics") else None
        fars_bw = idx[code]["metrics"]["body_words"] if idx[code]["metrics"] else None
        ratio = round(mt["body_words"] / fars_bw, 2) if mt and fars_bw else None
        row = dict(code=code, fars_type=ftype, arxiv=aid, title=r["title"], v1_year=r["v1_year"],
                   venue=r["venue"], venue_by=r["venue_by"], paper_type=ctype, type_source=type_src,
                   sim_tier=tier, tier_reason=reason, cite_roles=json.dumps(r["cite_roles"]),
                   relation=r["relation"], H1_human=r["H1_human"], H1_by=r["H1_by"],
                   H2_top_tier=r["H2_top_tier"], H3_type_match=h3, H4_source=r["H4_source"],
                   pangram_ai_frac=r["pangram_ai_frac"], iclr_rating=r["iclr_rating"], iclr_tier=r["iclr_tier"],
                   body_words=mt["body_words"] if mt else None, body_ratio=ratio,
                   appendix_words=mt["appendix_words"] if mt else None,
                   n_sections=mt["n_sections"] if mt else None, n_figures=mt["n_figures"] if mt else None,
                   n_tables=mt["n_tables"] if mt else None, n_unique_cites=mt["n_unique_cites"] if mt else None,
                   n_internal_refs=mt["n_internal_refs"] if mt else None,
                   has_method_fig=bool(c.get("method_fig_captions")) if c else None,
                   github=";".join(mt["github"][:2]) if mt else "",
                   tex_source=c["source"] if c else None, tex_root=c.get("tex_root") if c else None)
        row["hard_pass"] = (row["H1_human"] == "pass" and row["H2_top_tier"] == "pass"
                            and h3 == "pass" and row["H4_source"] == "pass")
        row["h4_pending"] = (row["H1_human"] == "pass" and row["H2_top_tier"] == "pass"
                             and h3 == "pass" and row["H4_source"] == "unknown")
        reasons = []
        if row["H1_human"] == "fail":
            reasons.append(f"H1:{r['H1_by']}")
        if row["H2_top_tier"] == "fail":
            reasons.append(f"H2:{r['venue']}")
        if h3 == "fail":
            reasons.append(f"H3:{ctype}!={ftype}")
        if r["H4_source"] == "fail":
            reasons.append("H4:" + ("short/unstructured tex" if mt else "no usable tex"))
        if tier is None and not reasons:
            reasons.append("not shortlisted for tier judging")
        row["reject_reason"] = ";".join(reasons)
        cands.append(row)
        rows_out.append(row)
    per_fars[code] = dict(fars_type=ftype, cands=cands)

# ---- ranking ----
def rank_key(r):
    return (-(r["sim_tier"] or 0), 0 if r["relation"] == "parent" else 1,
            0 if r["iclr_rating"] is not None else 1,
            0 if (r["body_ratio"] is not None and r["body_ratio"] <= RATIO_MAX) else 1,
            -((1 if r["github"] else 0) + (1 if r["has_method_fig"] else 0)),
            -(r["v1_year"] or 0), r["arxiv"])

dlq = set()
for code, d in per_fars.items():
    elig = [r for r in d["cands"] if (r["hard_pass"] or r["h4_pending"]) and (r["sim_tier"] or 0) >= 1]
    elig.sort(key=rank_key)
    d["ranked"] = elig
    for r in elig[:3]:
        if r["h4_pending"]:
            dlq.add(r["arxiv"])
json.dump(sorted(dlq), open(f"{OUTD}/cache/download_queue.json", "w"))
if dlq and "--force" not in sys.argv:
    print(f"{len(dlq)} top-3 candidates still need e-prints -> cache/download_queue.json; "
          f"run s7_download.py + s4_cand_metrics.py then rerun. (--force to assign anyway)")

# ---- global one-to-one assignment (fixed 5 reserved first) ----
taken = {}
for code, p in FIXED.items():
    if p.get("human_primary"):
        taken[p["human_primary"]["arxiv"]] = code
conflicts = []
assigned = {}
# deferred acceptance: every unassigned FARS proposes its best available candidate; on a clash the
# higher similarity tier wins (rule: on conflict, the higher tier wins), the loser proposes its next choice.
ptr = {code: 0 for code in per_fars}
def usable(code):
    L = per_fars[code]["ranked"]
    while ptr[code] < len(L):
        r = L[ptr[code]]
        if (r["h4_pending"] and r["arxiv"] in dlq) or r["arxiv"] in taken:
            ptr[code] += 1
            continue
        return r
    return None
active = set(per_fars)
while True:
    props = {}
    for code in sorted(active - set(assigned)):
        r = usable(code)
        if r is not None:
            props.setdefault(r["arxiv"], []).append((code, r))
    if not props:
        break
    for aid, plist in props.items():
        plist.sort(key=lambda cr: (-(cr[1]["sim_tier"] or 0), rank_key(cr[1]), cr[0]))
        wcode, wr = plist[0]
        taken[aid] = wcode
        assigned[wcode] = wr
        for code, r in plist[1:]:
            conflicts.append(dict(arxiv=aid, kept=wcode, kept_tier=wr["sim_tier"],
                                  bumped=code, bumped_tier=r["sim_tier"]))

# alternates: next ranked, not any assigned primary
pairs = []
for code in sorted(per_fars):
    d = per_fars[code]
    prim = assigned.get(code)
    alts = [r for r in d["ranked"] if prim and r["arxiv"] != prim["arxiv"] and r["arxiv"] not in taken
            and not (r["h4_pending"] and r["arxiv"] in dlq)][:2]
    fm = idx[code]["metrics"]
    def pack(r):
        return {k: r[k] for k in ("arxiv", "title", "v1_year", "venue", "paper_type", "sim_tier", "tier_reason",
                                  "relation", "H1_by", "pangram_ai_frac", "iclr_rating", "iclr_tier", "body_words",
                                  "body_ratio", "n_sections", "n_figures", "n_tables", "n_unique_cites",
                                  "has_method_fig", "github", "tex_source", "tex_root")}
    no_cand = None
    if prim is None:
        hard_ok = [r for r in d["cands"] if r["hard_pass"]]
        if not d["cands"]:
            no_cand = "empty candidate pool"
        elif not [r for r in d["cands"] if r["H1_human"] == "pass"]:
            no_cand = "no human-verified citation (all 2025+ unverified or Pangram>0.05)"
        elif not [r for r in d["cands"] if r["H1_human"] == "pass" and r["H2_top_tier"] == "pass"]:
            no_cand = "no top-tier accepted citation among human-verified"
        elif not hard_ok:
            no_cand = "hard filters leave none (mostly H3 type mismatch or H4)"
        elif not d["ranked"]:
            no_cand = "hard survivors are tier-0 tool/data citations only"
        else:
            no_cand = "all ranked candidates taken by higher-tier FARS or pending download"
    pairs.append(dict(code=code, fars_type=d["fars_type"],
                      ai=dict(title=idx[code]["title"],
                              body_words=fm["body_words"] if fm else None,
                              n_sections=fm["n_sections"] if fm else None,
                              n_figures=fm["n_figures"] if fm else None, n_tables=fm["n_tables"] if fm else None,
                              n_unique_cites=fm["n_unique_cites"] if fm else None,
                              n_internal_refs=fm["n_internal_refs"] if fm else None),
                      human_primary=pack(prim) if prim else None,
                      human_alternates=[pack(r) for r in alts],
                      no_candidate=(prim is None), no_candidate_reason=no_cand))

# frozen five, in the same output
for code in sorted(FIXED):
    p = FIXED[code]
    pairs.append(dict(code=code, fars_type=p["fars_type"], ai=p["ai"],
                      human_primary=p["human_primary"], human_alternates=p["human_alternates"],
                      no_candidate=False, no_candidate_reason=None, frozen_from="pairs5_0911_v3.json"))
pairs.sort(key=lambda p: p["code"])

json.dump(dict(generated="2026-09-11", rule="PAIR_RULE_0911.md (v3: type match is hard)",
               pool="papers cited by each FARS paper (candidates.json + analemma.bib + title lookups)",
               frozen_pairs=sorted(FIXED), n_fars=len(pairs),
               n_primary=sum(1 for p in pairs if p["human_primary"]),
               n_no_candidate=sum(1 for p in pairs if p["no_candidate"]),
               conflicts_resolved=conflicts, pairs=pairs),
          open(f"{OUTD}/pairs165_draft.json", "w"), indent=1, ensure_ascii=False)

# ---- csv over the full pool ----
fields = ["code", "fars_type", "arxiv", "title", "v1_year", "venue", "venue_by", "paper_type", "type_source",
          "sim_tier", "tier_reason", "relation", "cite_roles", "H1_human", "H1_by", "H2_top_tier",
          "H3_type_match", "H4_source", "hard_pass", "reject_reason", "pangram_ai_frac", "iclr_rating",
          "iclr_tier", "body_words", "body_ratio", "appendix_words", "n_sections", "n_figures", "n_tables",
          "n_unique_cites", "n_internal_refs", "has_method_fig", "github", "tex_source", "tex_root", "note"]
with open(f"{OUTD}/candidates_165.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows_out, key=lambda r: (r["code"], -(r.get("sim_tier") or -1), r.get("arxiv", ""))):
        w.writerow(r)

json.dump(dict(conflicts=conflicts, download_queue=sorted(dlq),
               assigned={c: r["arxiv"] for c, r in assigned.items()}),
          open(f"{OUTD}/cache/assign_report.json", "w"), indent=1, ensure_ascii=False)
print(f"FARS total {len(pairs)} | primary {sum(1 for p in pairs if p['human_primary'])} | "
      f"no_candidate {sum(1 for p in pairs if p['no_candidate'])} | conflicts {len(conflicts)} | "
      f"downloads pending {len(dlq)}")
