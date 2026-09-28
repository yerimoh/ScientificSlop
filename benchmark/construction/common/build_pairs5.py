"""Build the 5-pair SciSlopBench PoC manifest (FARS AI paper <-> human anchor) from cached inputs.

Inputs (same directory): fars5.json, cand_metrics.json, arxiv_meta.json, iclr26_records.json, iclr_nearest.json
Outputs (OUT): candidates_5fars_0911.csv, pairs5_0911.json
"""
import json, csv, re, os, sys
from datetime import date

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
FARS_DATE = date(2026, 2, 13)          # FARS live batch id live_20260213
ERA_MONTHS = 18
PANGRAM_MAX = 0.05
RATIO_MAX = 2.5

fars = json.load(open("fars5.json"))
cand = json.load(open("cand_metrics.json"))
am = json.load(open("arxiv_meta.json"))
iclr26 = json.load(open("iclr26_records.json"))

# ---- manual labels (read from abstracts / arXiv comments on 2026-09-11) ----
# paper type: method | framework | analysis | benchmark | survey | dataset
TYPE = {
 "2508.06249": "method", "2510.04340": "method", "2506.19823": "analysis", "2602.07852": "analysis",
 "2310.03693": "analysis", "2506.11613": "analysis", "2506.11618": "analysis", "2506.13206": "analysis",
 "2405.16833": "method", "2410.10014": "method", "2406.18495": "dataset", "2409.18169": "survey",
 "2412.14093": "analysis", "2106.09685": "method", "2203.02155": "method", "2312.03732": "method",
 "2401.05566": "analysis", "2502.17424": "analysis",
 "2509.22072": "method", "2510.01172": "method", "2509.24502": "method", "2410.02355": "method",
 "2502.11177": "benchmark", "2110.11309": "method", "2012.14913": "analysis", "2206.06520": "method",
 "2110.14168": "dataset", "2405.14768": "method", "2211.11031": "method", "2401.01286": "survey",
 "2210.07229": "method", "2009.03300": "benchmark", "1706.04115": "method",
 "2510.00615": "method", "2508.21433": "analysis", "2510.24699": "method", "2512.22087": "method",
 "2601.11868": "benchmark", "2510.12635": "method", "2402.02716": "survey", "2511.03690": "framework",
 "2509.23586": "method", "2210.03629": "method", "2405.15793": "method", "2310.06770": "benchmark",
 "2306.14898": "benchmark", "2412.15115": "framework", "2310.06839": "method",
 "2509.04499": "framework", "2508.15804": "benchmark", "2510.14240": "benchmark", "2310.11511": "method",
 "2506.01829": "framework", "2203.11147": "method", "2112.09332": "method", "2305.14251": "framework",
 "2303.08896": "method", "2004.04228": "framework", "2305.14627": "benchmark",
 "2407.18370": "method", "2504.08942": "benchmark", "2510.04040": "benchmark", "2201.11903": "method",
 "2203.11171": "method", "2306.05685": "benchmark", "2403.07718": "benchmark", "2307.13854": "benchmark",
 "2306.06070": "dataset", "2403.04132": "benchmark", "2202.05262": "method", "2303.11366": "method",
}
# accepted top-tier main venue (from arXiv comment / journal_ref / ICLR pool / OpenReview PoC memory)
VENUE = {
 "2508.06249": "ICML 2026", "2510.04340": "ICLR 2026 poster", "2506.19823": "ICLR 2026 poster",
 "2602.07852": "ICLR 2026 poster", "2310.03693": "ICLR 2024 oral", "2506.11613": "ICML 2025 (template)",
 "2506.11618": "ICML 2025 (template)", "2506.13206": "preprint", "2405.16833": "NeurIPS 2024",
 "2410.10014": "NeurIPS 2024 workshop", "2406.18495": "NeurIPS 2024 D&B", "2409.18169": "ACM CSUR",
 "2412.14093": "preprint", "2106.09685": "ICLR 2022", "2203.02155": "NeurIPS 2022", "2312.03732": "preprint",
 "2401.05566": "preprint", "2502.17424": "Nature 2026 / ICML 2025",
 "2509.22072": "ICLR 2026 poster", "2510.01172": "ICLR 2026 poster", "2509.24502": "ICLR 2026 poster",
 "2410.02355": "ICLR 2025 oral", "2502.11177": "ACL 2025", "2110.11309": "ICLR 2022", "2012.14913": "EMNLP 2021",
 "2206.06520": "ICML 2022", "2110.14168": "preprint", "2405.14768": "NeurIPS 2024", "2211.11031": "NeurIPS 2023",
 "2401.01286": "preprint", "2210.07229": "ICLR 2023", "2009.03300": "ICLR 2021", "1706.04115": "CoNLL 2017",
 "2510.00615": "ICML 2026 (ICLR 2026 reject)", "2508.21433": "DL4C workshop", "2510.24699": "ICLR 2026 poster",
 "2512.22087": "preprint", "2601.11868": "ICLR 2026 poster", "2510.12635": "preprint", "2402.02716": "preprint",
 "2511.03690": "MLSys 2026", "2509.23586": "FSE 2026", "2210.03629": "ICLR 2023", "2405.15793": "NeurIPS 2024",
 "2310.06770": "ICLR 2024 oral", "2306.14898": "NeurIPS 2023 D&B", "2412.15115": "tech report", "2310.06839": "ACL 2024",
 "2509.04499": "ICLR 2026 poster", "2508.15804": "ICLR 2026 reject", "2510.14240": "ICLR 2026 poster",
 "2310.11511": "ICLR 2024 oral", "2506.01829": "ACL 2025", "2203.11147": "preprint", "2112.09332": "preprint",
 "2305.14251": "EMNLP 2023", "2303.08896": "EMNLP 2023", "2004.04228": "ACL 2020", "2305.14627": "EMNLP 2023",
 "2407.18370": "ICLR 2025 oral", "2504.08942": "COLM 2025", "2510.04040": "ICLR 2026 poster", "2201.11903": "NeurIPS 2022",
 "2203.11171": "ICLR 2023", "2306.05685": "NeurIPS 2023 D&B", "2403.07718": "ICML 2024", "2307.13854": "ICLR 2024",
 "2306.06070": "NeurIPS 2023 D&B", "2403.04132": "ICML 2024", "2202.05262": "NeurIPS 2022", "2303.11366": "NeurIPS 2023",
}
TOP_TIER = ("ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "NAACL", "COLM", "Nature")

def accepted_top(v):
    if not v or "reject" in v.lower() and "(" not in v:
        return False
    if any(w in v for w in ("workshop", "preprint", "tech report", "template", "CoNLL", "CSUR", "MLSys", "FSE")):
        return False
    return any(t in v for t in TOP_TIER)

def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0

def topic_tie(f, aid):
    roles = f["cite_roles"].get(aid, {})
    if "experiments" in roles:
        return "parent"            # benchmark / protocol / baseline inherited by FARS
    if roles:
        return "cited"
    return "neighbor"

extra = {"FA0001": ["2506.19823", "2602.07852", "2510.04340"], "FA0209": ["2509.22072", "2510.01172", "2509.24502"],
         "FA0035": ["2601.11868", "2510.24699", "2508.21433", "2510.00615"], "FA0046": ["2508.15804", "2509.04499", "2510.14240"],
         "FA0006": ["2504.08942", "2510.04040"]}

rows = []
for code, f in fars.items():
    fm = f["metrics"]
    ids = sorted(set(f["cited_arxiv"]) | set(extra[code]))
    for aid in ids:
        a = am.get(aid)
        cm = cand.get(aid, {}).get("metrics")
        if not a or not cm:
            rows.append(dict(code=code, arxiv=aid, title=(a or {}).get("title", ""), note="no e-print"))
            continue
        pub = date.fromisoformat(a["published"][:10])
        rec = iclr26.get(aid)
        venue = VENUE.get(aid, "")
        ratio = cm["body_words"] / fm["body_words"]
        pangram = rec["fraction_ai"] if rec else None
        r = dict(
            code=code, arxiv=aid, title=a["title"], v1_date=a["published"][:10], venue=venue,
            paper_type=TYPE.get(aid, "?"), topic_tie=topic_tie(f, aid), cite_roles=json.dumps(f["cite_roles"].get(aid, {})),
            iclr_rating=rec["rating_mean"] if rec else None, iclr_tier=rec["tier"] if rec else None,
            pangram_ai_frac=pangram,
            body_words=cm["body_words"], body_ratio=round(ratio, 2), appendix_words=cm["appendix_words"],
            n_sections=cm["n_sections"], n_figures=cm["n_figures"], n_tables=cm["n_tables"],
            n_unique_cites=cm["n_unique_cites"], n_internal_refs=cm["n_internal_refs"],
            has_method_fig=bool(cand[aid].get("method_fig_captions")), github=";".join(cm["github"][:2]),
            template=";".join(cm["template"][:1]),
            # hard filters
            H1_human_verified=("pass" if (pangram is not None and pangram <= PANGRAM_MAX) or pub < date(2022, 11, 30)
                               else ("fail" if pangram is not None else "unknown")),
            H2_accepted_top_tier="pass" if accepted_top(venue) else "fail",
            H3_era_18mo="pass" if 0 <= months_between(pub, FARS_DATE) <= ERA_MONTHS else "fail",
            H4_tex_ok="pass" if cm["body_words"] >= 2500 and cm["n_sections"] >= 4 else "fail",
            # soft
            S1_type_match=1 if TYPE.get(aid) == "method" else (0.5 if TYPE.get(aid) in ("framework", "analysis") else 0),
            S2_topic={"parent": 2, "cited": 1, "neighbor": 0.5}[topic_tie(f, aid)],
            S3_score_available=1 if rec else 0,
            S4_scope=1 if ratio <= RATIO_MAX else 0,
            S5_artifacts=(0.5 if cm["github"] else 0) + (0.5 if cand[aid].get("method_fig_captions") else 0),
        )
        hard = [r["H1_human_verified"], r["H2_accepted_top_tier"], r["H3_era_18mo"], r["H4_tex_ok"]]
        r["hard_pass"] = sum(1 for h in hard if h == "pass")
        r["hard_fail"] = sum(1 for h in hard if h == "fail")
        r["soft_score"] = r["S1_type_match"] * 2 + r["S2_topic"] + r["S3_score_available"] + r["S4_scope"] + r["S5_artifacts"]
        rows.append(r)

keys = [k for k in rows[-1].keys()]
with open(os.path.join(OUT, "candidates_5fars_0911.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys + ["note"], extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["code"], -(r.get("hard_pass", -1)), -(r.get("soft_score", -1)))):
        w.writerow(r)

# ---- decisions (human judgment on top of the rubric; rationale in PAIRS_README_0911.md) ----
DECISION = {
 "FA0001": dict(primary="2510.04340", alternates=["2508.06249", "2506.19823"], legacy_poc="2310.03693"),
 "FA0209": dict(primary="2509.22072", alternates=["2510.01172", "2410.02355"], legacy_poc="2410.02355"),
 "FA0035": dict(primary="2510.00615", alternates=["2601.11868", "2508.21433"], legacy_poc="2310.06770"),
 "FA0046": dict(primary="2510.14240", alternates=["2509.04499", "2506.01829"], legacy_poc="2310.11511"),
 "FA0006": dict(primary="2407.18370", alternates=["2504.08942"], legacy_poc="2407.18370"),
}
byid = {(r["code"], r["arxiv"]): r for r in rows if "body_words" in r}
pairs = []
for code, d in DECISION.items():
    f = fars[code]
    def pack(aid):
        r = byid[(code, aid)]
        src = "eprints_new" if cand[aid].get("source") == "eprints_new" else "human_refs/eprints"
        return dict(arxiv=aid, title=r["title"], v1_date=r["v1_date"], venue=r["venue"], paper_type=r["paper_type"],
                    topic_tie=r["topic_tie"], iclr_rating=r["iclr_rating"], iclr_tier=r["iclr_tier"], pangram_ai_frac=r["pangram_ai_frac"],
                    body_words=r["body_words"], body_ratio=r["body_ratio"], appendix_words=r["appendix_words"],
                    n_figures=r["n_figures"], n_tables=r["n_tables"], n_unique_cites=r["n_unique_cites"], n_internal_refs=r["n_internal_refs"],
                    has_method_fig=r["has_method_fig"], github=r["github"], template=r["template"],
                    hard=dict(H1=r["H1_human_verified"], H2=r["H2_accepted_top_tier"], H3=r["H3_era_18mo"], H4=r["H4_tex_ok"]),
                    soft_score=r["soft_score"], tex_source=f"{src}/{aid}", tex_root=cand[aid]["tex_root"])
    fm = f["metrics"]
    pairs.append(dict(
        code=code, ai=dict(title=f["title"], dir=os.path.relpath(f["dir"], os.environ.get("SCISLOP_ROOT", ".")),
                           body_words=fm["body_words"], n_sections=fm["n_sections"], n_figures=fm["n_figures"], n_tables=fm["n_tables"],
                           n_unique_cites=fm["n_unique_cites"], n_internal_refs=fm["n_internal_refs"],
                           has_method_diagram=f["has_method_diagram"], has_code=f["has_code"], has_results=f["has_results"]),
        human_primary=pack(d["primary"]), human_alternates=[pack(a) for a in d["alternates"]],
        legacy_poc_anchor=pack(d["legacy_poc"])))
json.dump(dict(generated="2026-09-11", fars_date=str(FARS_DATE), rules=dict(era_months=ERA_MONTHS, pangram_max=PANGRAM_MAX, ratio_max=RATIO_MAX),
               pairs=pairs), open(os.path.join(OUT, "pairs5_0911.json"), "w"), indent=1, ensure_ascii=False)

for p in pairs:
    h = p["human_primary"]
    print(f"{p['code']}  AI body {p['ai']['body_words']:5d} | HU {h['arxiv']} body {h['body_words']} x{h['body_ratio']} {h['venue']} rating={h['iclr_rating']} pangram={h['pangram_ai_frac']} type={h['paper_type']} tie={h['topic_tie']} hard={h['hard']} soft={h['soft_score']} | {h['title'][:60]}")
