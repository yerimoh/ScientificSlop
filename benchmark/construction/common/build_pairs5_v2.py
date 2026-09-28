"""SciSlopBench 5-pair manifest, rule set v2 (user revision, 2026-09-11).

v2 rules
  Pool  : papers the FARS paper cites (related work, baselines, benchmarks), arXiv e-print available.
  H2    : accepted at a top-tier main venue (from the original request); rejected-only / workshop / preprint excluded.
  H1    : human-written verified = (ICLR 2026 submission with Pangram AI fraction <= 0.05) OR (arXiv v1 year <= 2024).
  Rank  : 1) model-judged topic similarity (tier 3 same problem+contribution kind / 2 same problem / 1 same area / 0 tool only;
             SPECTER2 cosine breaks ties inside a tier)
          2) paper type match (method > framework/analysis > benchmark/survey/dataset)
          3) citation relation (experiments = parent > related work / intro = cited)
          4) review score available (ICLR 2026 record)      5) body-word ratio <= 2.5      6) code repo + method figure
Outputs: candidates_5fars_0911_v2.csv, pairs5_0911_v2.json
"""
import json, csv, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
PANGRAM_MAX, RATIO_MAX = 0.05, 2.5
fars = json.load(open("fars5.json")); cand = json.load(open("cand_metrics.json")); am = json.load(open("arxiv_meta.json"))
iclr26 = json.load(open("iclr26_records.json")); sim = json.load(open("embed_sim.json"))
sys.path.insert(0, ".")
from build_pairs5 import TYPE, VENUE, accepted_top   # manual labels from v1
# OpenReview ratings for ICLR 2025 anchors (HF dataset QAQqaq/ICLR2025Openreview, fetched 2026-09-11)
OR_RATING = {"2410.02355": dict(rating_mean=8.0, ratings=[8, 8, 8, 8], venue="ICLR 2025 oral"),
             "2407.18370": dict(rating_mean=8.0, ratings=[10, 8, 8, 6], venue="ICLR 2025 oral")}

TYPE.setdefault("2305.14627", "benchmark")
VENUE.setdefault("2305.14627", "EMNLP 2023")
# citations whose bib entry has no arXiv id (mapped by hand)
MANUAL_CITED = {"FA0035": {"2601.11868": {"experiments": 1, "introduction": 1, "related_work": 1}},
                "FA0046": {"2305.14627": {"related_work": 1}}}
# model-judged similarity tier (Claude reading of title+abstract vs FARS title+abstract)
TIER = {
 "FA0001": {"2508.06249": 3, "2405.16833": 3, "2410.10014": 3, "2310.03693": 2, "2506.19823": 2, "2502.17424": 2, "2506.11613": 2,
            "2506.11618": 2, "2409.18169": 2, "2506.13206": 1, "2406.18495": 1, "2412.14093": 1, "2401.05566": 1, "2203.02155": 1,
            "2312.03732": 1, "2106.09685": 1},
 "FA0209": {"2509.22072": 3, "2410.02355": 2, "2206.06520": 2, "2110.11309": 2, "2211.11031": 2, "2202.05262": 2, "2210.07229": 2,
            "2405.14768": 2, "2502.11177": 2, "2401.01286": 2, "2012.14913": 1, "2110.14168": 0, "2009.03300": 0, "1706.04115": 0},
 "FA0035": {"2510.00615": 3, "2512.22087": 3, "2508.21433": 3, "2509.23586": 3, "2510.12635": 3, "2601.11868": 2, "2306.14898": 1,
            "2402.02716": 1, "2210.03629": 1, "2412.15115": 0, "2405.15793": 1, "2310.06770": 1, "2511.03690": 1},
 "FA0046": {"2508.15804": 3, "2506.01829": 3, "2305.14627": 3, "2203.11147": 3, "2305.14251": 2, "2310.11511": 2, "2303.08896": 2,
            "2112.09332": 2, "2004.04228": 1},
 "FA0006": {"2504.08942": 3, "2407.18370": 3, "2306.05685": 2, "2203.11171": 2, "2307.13854": 1, "2306.06070": 1, "2403.07718": 1,
            "2403.04132": 1},
}
TYPE_SCORE = {"method": 2, "framework": 1, "analysis": 1, "benchmark": 0, "survey": 0, "dataset": 0}

rows = []
for code, f in fars.items():
    roles = dict(f["cite_roles"]); roles.update(MANUAL_CITED.get(code, {}))
    for aid, rl in roles.items():
        a = am.get(aid); cm = cand.get(aid, {}).get("metrics")
        if not a:
            continue
        yr = int(a["published"][:4]); rec = iclr26.get(aid); pg = rec["fraction_ai"] if rec else None
        if pg is not None and pg <= PANGRAM_MAX:
            h1, h1_by = "pass", "pangram"
        elif yr <= 2024:
            h1, h1_by = "pass", "year<=2024"
        else:
            h1, h1_by = "fail", ("pangram>0.05" if pg is not None else "2025+ unverified")
        h2 = "pass" if accepted_top(VENUE.get(aid, "")) else "fail"
        orr = OR_RATING.get(aid)
        r = dict(code=code, arxiv=aid, title=a["title"], v1_year=yr, venue=VENUE.get(aid, ""), paper_type=TYPE.get(aid, "?"), H2_top_tier=h2,
                 cite_roles=json.dumps(rl), relation="parent" if "experiments" in rl else "cited",
                 sim_tier=TIER[code].get(aid, 1), specter2=sim[code]["specter2"].get(aid), bge=sim[code]["bge"].get(aid),
                 H1=h1, H1_by=h1_by, iclr_rating=(rec["rating_mean"] if rec else (orr["rating_mean"] if orr else None)),
                 iclr_tier=(rec["tier"] if rec else (orr["venue"] if orr else None)), rating_source=("iclr26_pool" if rec else ("hf_QAQqaq_ICLR2025" if orr else None)), pangram_ai_frac=pg,
                 eprint=bool(cm))
        if cm:
            ratio = cm["body_words"] / f["metrics"]["body_words"]
            r.update(body_words=cm["body_words"], body_ratio=round(ratio, 2), appendix_words=cm["appendix_words"], n_figures=cm["n_figures"],
                     n_tables=cm["n_tables"], n_unique_cites=cm["n_unique_cites"], n_internal_refs=cm["n_internal_refs"],
                     has_method_fig=bool(cand[aid].get("method_fig_captions")), github=";".join(cm["github"][:2]), template=";".join(cm["template"][:1]))
            r["rank_key"] = (-r["sim_tier"], -TYPE_SCORE.get(r["paper_type"], 0), -(2 if r["relation"] == "parent" else 1),
                             -(1 if (rec or orr) else 0), -(1 if ratio <= RATIO_MAX else 0), -((1 if cm["github"] else 0) + (1 if r["has_method_fig"] else 0)),
                             -(r["specter2"] or 0))
        rows.append(r)

for code in fars:
    R = [r for r in rows if r["code"] == code and r["eprint"] and r["H1"] == "pass" and r["H2_top_tier"] == "pass"]
    R.sort(key=lambda r: r["rank_key"])
    for i, r in enumerate(R):
        r["v2_rank"] = i + 1

fields = [k for k in rows[0].keys() if k != "rank_key"] + ["v2_rank"]
with open(os.path.join(OUT, "candidates_5fars_0911_v2.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
    for r in sorted(rows, key=lambda r: (r["code"], r.get("v2_rank", 99), -(r["specter2"] or 0))):
        w.writerow(r)

byid = {(r["code"], r["arxiv"]): r for r in rows}
pairs = []
for code, f in fars.items():
    ranked = [r for r in rows if r["code"] == code and r.get("v2_rank")]
    ranked.sort(key=lambda r: r["v2_rank"])
    top_fail = sorted([r for r in rows if r["code"] == code and (r["H1"] == "fail" or r["H2_top_tier"] == "fail") and r["eprint"]], key=lambda r: (-r["sim_tier"], -(r["specter2"] or 0)))[:3]
    def pack(r):
        src = "eprints_new" if cand[r["arxiv"]].get("source") == "eprints_new" else "human_refs/eprints"
        d = {k: v for k, v in r.items() if k not in ("rank_key", "code")}
        d["tex_source"] = f"{src}/{r['arxiv']}"; d["tex_root"] = cand[r["arxiv"]].get("tex_root"); return d
    fm = f["metrics"]
    pairs.append(dict(code=code, ai=dict(title=f["title"], body_words=fm["body_words"], n_sections=fm["n_sections"], n_figures=fm["n_figures"],
                                         n_tables=fm["n_tables"], n_unique_cites=fm["n_unique_cites"], n_internal_refs=fm["n_internal_refs"]),
                      human_primary=pack(ranked[0]), human_alternates=[pack(r) for r in ranked[1:3]],
                      excluded_by_H1_H2_but_most_similar=[pack(r) for r in top_fail]))
json.dump(dict(generated="2026-09-11", rule_set="v2 (user revision)", pool="cited papers only", H1="ICLR2026 Pangram<=0.05 or v1 year<=2024", H2="accepted at a top-tier main venue (original request: top-tier papers only)",
               ranking=["sim_tier (model judged; SPECTER2 tiebreak)", "paper_type method>framework/analysis>benchmark", "relation parent>cited",
                        "ICLR score available", "body_ratio<=2.5", "code+method figure"], pairs=pairs),
          open(os.path.join(OUT, "pairs5_0911_v2.json"), "w"), indent=1, ensure_ascii=False)
for p in pairs:
    h = p["human_primary"]
    print(f"{p['code']} -> {h['arxiv']} tier={h['sim_tier']} sp={h['specter2']:.3f} {h['paper_type']:9s} {h['relation']:6s} {h['venue']:<30} rate={h['iclr_rating']} H1={h['H1_by']:<11} x{h['body_ratio']} | {h['title'][:60]}")
    for r in p["human_alternates"]:
        print(f"      alt {r['arxiv']} tier={r['sim_tier']} sp={r['specter2']:.3f} {r['paper_type']:9s} {r['relation']:6s} {r['venue']:<30} x{r.get('body_ratio')} | {r['title'][:50]}")
    for r in p["excluded_by_H1_H2_but_most_similar"]:
        print(f"      excl {r['arxiv']} tier={r['sim_tier']} sp={r['specter2']:.3f} {r['paper_type']:9s} H1={r['H1_by']} H2={r['H2_top_tier']} {r['venue']} | {r['title'][:50]}")
