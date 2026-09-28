"""Step 6 (Agents4Science): final H3 with the judged types, ranking (PAIR_RULE_0911 step 3),
one-to-one assignment across the corpus, and outputs.

Candidates with H4=unknown that land in a paper's top 3 go to cache/download_queue.json; run
a7_download.py + a4_cand_metrics.py, then rerun this script.

Outputs: ../Agents4Science/candidates_a4s.csv, pairs_a4s.json, cache/assign_report.json
"""
import os, re, csv, json, glob, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"
RATIO_MAX = 2.5

idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
filters = json.load(open(f"{OUTD}/cache/filters.json"))
shortlists = json.load(open(f"{OUTD}/cache/shortlists.json"))
cm = json.load(open(f"{OUTD}/cache/cand_metrics_a4s.json"))
meta = json.load(open(f"{OUTD}/cache/arxiv_meta_a4s.json"))
types = json.load(open(f"{OUTD}/cache/a4s_types.json"))          # step 0, read from title+abstract

judged = {}
for f in sorted(glob.glob(f"{OUTD}/cache/judge_out/batch_*.json")):
    for rec in json.load(open(f)):
        judged.setdefault(rec["code"], {}).update(rec.get("candidates", {}))

rows_out, per_paper = [], {}
for code, srows in filters.items():
    ftype = types[code]
    j = judged.get(code, {})
    cands = []
    for aid, r in srows.items():
        if r.get("error"):
            rows_out.append(dict(code=code, arxiv=aid, note=r["error"]))
            continue
        jc = j.get(aid)
        ctype = jc["type"] if jc else r["heur_type"]
        tier = jc["tier"] if jc else None
        row = dict(code=code, a4s_type=ftype, arxiv=aid, title=r["title"], v1_year=r["v1_year"],
                   venue=r["venue"], venue_by=r["venue_by"], paper_type=ctype,
                   type_source="judged" if jc else "heuristic", sim_tier=tier,
                   tier_reason=jc["reason"] if jc else "", cite_roles=json.dumps(r["cite_roles"]),
                   relation=r["relation"], id_source=r["id_source"],
                   H1_human=r["H1_human"], H1_by=r["H1_by"], H2_top_tier=r["H2_top_tier"],
                   H3_type_match="pass" if ctype == ftype else "fail", H4_source=r["H4_source"],
                   pangram_ai_frac=r["pangram_ai_frac"], iclr_rating=r["iclr_rating"],
                   iclr_tier=r["iclr_tier"], oa_citations=r.get("oa_citations"))
        c = cm.get(aid)
        mt = c["metrics"] if c and c.get("metrics") else None
        a4s_bw = idx[code]["metrics"]["body_words"]
        row.update(body_words=mt["body_words"] if mt else None,
                   body_ratio=round(mt["body_words"] / a4s_bw, 2) if mt and a4s_bw else None,
                   appendix_words=mt["appendix_words"] if mt else None,
                   n_sections=mt["n_sections"] if mt else None, n_figures=mt["n_figures"] if mt else None,
                   n_tables=mt["n_tables"] if mt else None, n_unique_cites=mt["n_unique_cites"] if mt else None,
                   n_internal_refs=mt["n_internal_refs"] if mt else None,
                   has_method_fig=bool(c.get("method_fig_captions")) if c else None,
                   github=";".join(mt["github"][:2]) if mt else "",
                   template=";".join(mt["template"][:1]) if mt else "",
                   tex_source=(c["source"] + "/" + aid) if c else None,
                   tex_root=c.get("tex_root") if c else None)
        row["hard_pass"] = (row["H1_human"] == "pass" and row["H2_top_tier"] == "pass"
                            and row["H3_type_match"] == "pass" and row["H4_source"] == "pass")
        row["h4_pending"] = (row["H1_human"] == "pass" and row["H2_top_tier"] == "pass"
                             and row["H3_type_match"] == "pass" and row["H4_source"] == "unknown")
        reasons = []
        if row["H1_human"] == "fail":
            reasons.append(f"H1:{r['H1_by']}")
        if row["H2_top_tier"] == "fail":
            reasons.append(f"H2:{r['venue']}")
        if row["H3_type_match"] == "fail":
            reasons.append(f"H3:{ctype}!={ftype}")
        if r["H4_source"] == "fail":
            reasons.append("H4:" + ("short/unstructured tex" if mt else "no usable tex"))
        if tier is None and not reasons:
            reasons.append("not shortlisted for tier judging")
        row["reject_reason"] = ";".join(reasons)
        cands.append(row)
        rows_out.append(row)
    per_paper[code] = dict(a4s_type=ftype, cands=cands)


def rank_key(r):
    return (-(r["sim_tier"] or 0), 0 if r["relation"] == "parent" else 1,
            0 if r["iclr_rating"] is not None else 1,
            0 if (r["body_ratio"] is not None and r["body_ratio"] <= RATIO_MAX) else 1,
            -((1 if r["github"] else 0) + (1 if r["has_method_fig"] else 0)),
            -(r["v1_year"] or 0), r["arxiv"])


dlq = set()
for code, d in per_paper.items():
    elig = [r for r in d["cands"] if (r["hard_pass"] or r["h4_pending"]) and (r["sim_tier"] or 0) >= 1]
    elig.sort(key=rank_key)
    d["ranked"] = elig
    for r in elig[:3]:
        if r["h4_pending"]:
            dlq.add(r["arxiv"])
json.dump(sorted(dlq), open(f"{OUTD}/cache/download_queue.json", "w"))
if dlq and "--force" not in sys.argv:
    print(f"{len(dlq)} top-3 candidates still need e-prints -> cache/download_queue.json; "
          f"run a7_download.py + a4_cand_metrics.py then rerun. (--force to assign anyway)")

# ---- one-to-one assignment, deferred acceptance, higher tier wins a clash ----
taken, assigned, conflicts = {}, {}, []
ptr = {code: 0 for code in per_paper}


def usable(code):
    L = per_paper[code]["ranked"]
    while ptr[code] < len(L):
        r = L[ptr[code]]
        if (r["h4_pending"] and r["arxiv"] in dlq) or r["arxiv"] in taken:
            ptr[code] += 1
            continue
        return r
    return None


while True:
    props = {}
    for code in sorted(set(per_paper) - set(assigned)):
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
        for c2, r2 in plist[1:]:
            conflicts.append(dict(arxiv=aid, kept=wcode, kept_tier=wr["sim_tier"],
                                  bumped=c2, bumped_tier=r2["sim_tier"]))

PACK = ("arxiv", "title", "v1_year", "venue", "paper_type", "sim_tier", "tier_reason", "relation",
        "cite_roles", "H1_human", "H1_by", "H2_top_tier", "H3_type_match", "H4_source",
        "pangram_ai_frac", "iclr_rating", "iclr_tier", "body_words", "body_ratio", "appendix_words",
        "n_sections", "n_figures", "n_tables", "n_unique_cites", "n_internal_refs",
        "has_method_fig", "github", "template", "tex_source", "tex_root")

pairs = []
for code in sorted(per_paper):
    d = per_paper[code]
    prim = assigned.get(code)
    alts = [r for r in d["ranked"] if prim and r["arxiv"] != prim["arxiv"] and r["arxiv"] not in taken
            and not (r["h4_pending"] and r["arxiv"] in dlq)][:2]
    blocked = sorted([r for r in d["cands"] if not r["hard_pass"] and (r["sim_tier"] or 0) >= 2],
                     key=lambda r: (-(r["sim_tier"] or 0), r["arxiv"]))[:2]
    fm = idx[code]["metrics"]
    no_cand = None
    if prim is None:
        if not d["cands"]:
            no_cand = "empty candidate pool"
        elif not [r for r in d["cands"] if r["H1_human"] == "pass"]:
            no_cand = "no human-verified reference (all 2025+ unverified or Pangram>0.05)"
        elif not [r for r in d["cands"] if r["H1_human"] == "pass" and r["H2_top_tier"] == "pass"]:
            no_cand = "no top-tier accepted reference among human-verified"
        elif not [r for r in d["cands"] if r["hard_pass"]]:
            no_cand = "hard filters leave none (mostly H3 type mismatch or H4)"
        elif not d["ranked"]:
            no_cand = "hard survivors are tier-0 tool/data citations only"
        else:
            no_cand = "all ranked candidates taken by a higher-tier paper or pending download"
    pairs.append(dict(code=code, submission_id=idx[code]["submission_id"],
                      submission_number=idx[code]["submission_number"],
                      a4s_status=idx[code]["status"], a4s_type=d["a4s_type"],
                      a4s_scores=idx[code].get("scores"), a4s_autonomy=idx[code].get("autonomy"),
                      pool="cited",
                      ai=dict(title=idx[code]["title"], body_words=fm["body_words"],
                              n_sections=fm["n_sections"], n_figures=fm["n_figures"],
                              n_tables=fm["n_tables"], n_refs=fm["n_refs"],
                              n_unique_cites=fm["n_unique_cites"], n_internal_refs=fm["n_internal_refs"],
                              appendix_words=fm["appendix_words"],
                              tex_in_supplementary=idx[code]["tex_in_supplementary"],
                              n_tex_files=idx[code]["n_tex_files"],
                              pdf=os.path.join(idx[code]["dir"], "paper.pdf")),
                      human_primary={k: prim[k] for k in PACK} if prim else None,
                      human_alternates=[{k: r[k] for k in PACK} for r in alts],
                      blocked_but_most_similar=[dict({k: r[k] for k in PACK},
                                                     blocked_by=r["reject_reason"]) for r in blocked],
                      no_candidate=(prim is None), no_candidate_reason=no_cand))

json.dump(dict(generated="2026-09-11", corpus="Agents4Science 2025, AI/ML track (154 papers)",
               rule="PAIR_RULE_0911.md (v3: paper type is a hard filter)",
               pool="papers cited by each A4S paper (reference list parsed from the PDF, "
                    "titles resolved on arXiv)",
               deviations=["the AI side has no TeX, so its structural metrics come from the PDF text "
                           "(see scripts/a1_a4s_index.py); they are not interchangeable with the "
                           "TeX-derived numbers on the human side",
                           "H2 has no candidates.json equivalent, so it rests on the arXiv comment / "
                           "journal_ref and, where available, the OpenAlex publication venue"],
               n_papers=len(pairs), n_primary=sum(1 for p in pairs if p["human_primary"]),
               n_no_candidate=sum(1 for p in pairs if p["no_candidate"]),
               conflicts_resolved=conflicts, pairs=pairs),
          open(f"{OUTD}/pairs_a4s.json", "w"), indent=1, ensure_ascii=False)

fields = ["code", "a4s_type", "arxiv", "title", "v1_year", "venue", "venue_by", "paper_type",
          "type_source", "sim_tier", "tier_reason", "relation", "cite_roles", "id_source",
          "H1_human", "H1_by", "H2_top_tier", "H3_type_match", "H4_source", "hard_pass",
          "reject_reason", "pangram_ai_frac", "iclr_rating", "iclr_tier", "oa_citations",
          "body_words", "body_ratio", "appendix_words", "n_sections", "n_figures", "n_tables",
          "n_unique_cites", "n_internal_refs", "has_method_fig", "github", "template",
          "tex_source", "tex_root", "note"]
with open(f"{OUTD}/candidates_a4s.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows_out, key=lambda r: (r["code"], -(r.get("sim_tier") or -1), r.get("arxiv", ""))):
        w.writerow(r)

json.dump(dict(conflicts=conflicts, download_queue=sorted(dlq),
               assigned={c: r["arxiv"] for c, r in assigned.items()}),
          open(f"{OUTD}/cache/assign_report.json", "w"), indent=1, ensure_ascii=False)
print(f"papers {len(pairs)} | primary {sum(1 for p in pairs if p['human_primary'])} | "
      f"no_candidate {sum(1 for p in pairs if p['no_candidate'])} | conflicts {len(conflicts)} | "
      f"downloads pending {len(dlq)}")
