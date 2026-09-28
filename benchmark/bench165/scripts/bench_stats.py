"""Benchmark statistics for the 143 SciSlopBench pairs (0912, meeting A8).

Reads the pair manifest, the candidate table, the FARS index and the bench items and
writes results/bench_stats_0912.json. Every number in the paper's benchmark-statistics
tables comes from this file. The SPECTER2 alignment audit is a separate script
(bench_stats_specter_align.py) whose output this script only copies in.
Topic labels are one annotator's title-based labels (results/bench_stats_topic_labels.json).
"""
import os
import json, csv, glob, collections, statistics as st, re
ROOT = os.environ.get("SCISLOP_ROOT", ".")
SB = f"{ROOT}/paper/draft_v6/scislopbench"
D = json.load(open(f"{SB}/data/pairs165_0911/pairs165_draft.json"))
ITEMS = json.load(open(f"{SB}/bench165/items165.json"))
FI = json.load(open(f"{SB}/data/pairs165_0911/cache/fars_index.json"))
CM = json.load(open(f"{SB}/data/pairs165_0911/cache/cand_metrics_165.json"))
ROWS = [r for r in csv.DictReader(open(f"{SB}/data/pairs165_0911/candidates_165.csv")) if r["arxiv"]]
TOPIC = json.load(open(f"{SB}/bench165/results/bench_stats_topic_labels.json"))
ALIGN = json.load(open(f"{SB}/bench165/results/bench_stats_specter_align.json"))
WS = json.load(open(f"{SB}/data/pairs165_0911/cache/widen_shortlist.json"))

bench = {i["pair"] for i in ITEMS["items"]}
allp = D["pairs"]
prim = [p for p in allp if p.get("human_primary")]
B = [p for p in prim if p["code"] in bench]
out = {"n_fars": len(allp), "n_primary": len(prim), "n_bench": len(B)}

# ---- FARS source corpus ----
metas = [json.load(open(f)) for f in glob.glob(f"{ROOT}/fars/papers/*/meta.json")]
ts = [m["gmt_create"] for m in metas if m.get("gmt_create")]
import datetime
out["fars"] = dict(
    n=len(metas),
    created_min=datetime.datetime.fromtimestamp(min(ts) / 1000, datetime.UTC).date().isoformat(),
    created_max=datetime.datetime.fromtimestamp(max(ts) / 1000, datetime.UTC).date().isoformat(),
    code_clone=sum(bool(m.get("downloads", {}).get("code_clone")) for m in metas),
    traces=sum(bool(m.get("downloads", {}).get("traces")) for m in metas),
    review_json=sum(bool(m.get("downloads", {}).get("review_json")) for m in metas),
    latex=sum(1 for c in FI if FI[c].get("metrics")),
    no_latex=[c for c in FI if not FI[c].get("metrics")],
    type_all=dict(collections.Counter(p["fars_type"] for p in allp)),
    type_bench=dict(collections.Counter(p["fars_type"] for p in B)),
    type_unassigned=dict(collections.Counter(p["fars_type"] for p in allp if not p.get("human_primary"))),
    unassigned_reason=dict(collections.Counter(p.get("no_candidate_reason") for p in allp if not p.get("human_primary"))),
)

# ---- candidate funnel (cited pool, 161 FARS outside the five frozen pairs) ----
f = {"rows": len(ROWS), "fars": len({r["code"] for r in ROWS}), "unique_papers": len({r["arxiv"] for r in ROWS}),
     "median_cands_per_fars": st.median(collections.Counter(r["code"] for r in ROWS).values())}
s = ROWS
for col in ["H1_human", "H2_top_tier", "H3_type_match", "H4_source"]:
    s = [r for r in s if r[col] == "pass"]
    f["after_" + col] = len(s); f["fars_after_" + col] = len({r["code"] for r in s})
f["fail_any"] = {col: sum(1 for r in ROWS if r[col] == "fail") for col in ["H1_human", "H2_top_tier", "H3_type_match", "H4_source"]}
f["H4_unknown"] = sum(1 for r in ROWS if r["H4_source"] == "unknown")
f["hard_pass"] = sum(1 for r in ROWS if r["hard_pass"] == "True")
f["H1_fail_by"] = dict(collections.Counter(r["H1_by"] for r in ROWS if r["H1_human"] == "fail"))
f["H3_fail_by_fars_type"] = dict(collections.Counter(r["fars_type"] for r in ROWS if r["H3_type_match"] == "fail"))
f["cand_type_judged"] = dict(collections.Counter(r["paper_type"] for r in ROWS if r["type_source"] == "judged"))
f["parent_rows"] = sum(1 for r in ROWS if r["relation"] == "parent")
f["parent_hard_pass"] = sum(1 for r in ROWS if r["relation"] == "parent" and r["hard_pass"] == "True")
f["conflicts_resolved"] = len(D["conflicts_resolved"])
f["frozen_pairs"] = D["frozen_pairs"]
f["widen_shortlisted_fars"] = len(WS)
f["widened_assigned"] = len(D["widened"])
# widening pool
n_pool = 0
for line in open(f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl"):
    r = json.loads(line); fai = r.get("fraction_ai")
    if r.get("accept") in (True, "True", "true") and fai not in (None, "None") and float(fai) <= 0.05 and r.get("arxiv_id") not in (None, "", "None"):
        n_pool += 1
f["widen_pool"] = n_pool
out["funnel"] = f

# ---- composition of the 143 pairs ----
H = []
for p in B:
    h = dict(p["human_primary"]); c = CM.get(h["arxiv"])
    if c and c.get("metrics"):
        for k, v in c["metrics"].items(): h.setdefault(k, v)
        h.setdefault("has_method_fig", bool(c.get("method_fig_captions")))
    H.append(h)
A = [FI[p["code"]]["metrics"] for p in B]
def fam(v):
    v = v.replace("(ICLR 2026 reject)", "")
    for n in ["ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "NAACL", "CVPR", "ICCV", "ECCV", "COLM", "TMLR", "CCS"]:
        if n in v: return "ACL" if n == "NAACL" else n
    return "COLM" if "Language Model" in v else v
def v1(a): return 2000 + int(a[:2])
comp = dict(
    tier=dict(collections.Counter(h["sim_tier"] for h in H)),
    tier_ge2=sum(1 for h in H if h["sim_tier"] >= 2),
    pool=dict(collections.Counter(h.get("pool", "cited") for h in H)),
    relation=dict(collections.Counter(h["relation"] for h in H)),
    tier_by_pool={f"{h.get('pool','cited')}|{h['sim_tier']}": 0 for h in H},
    venue_family=dict(collections.Counter(fam(h["venue"]) for h in H).most_common()),
    venue_award=dict(collections.Counter(h["venue"] for h in H if "oral" in h["venue"].lower() or "spotlight" in h["venue"].lower())),
    v1_year=dict(sorted(collections.Counter(v1(h["arxiv"]) for h in H).items())),
    verification=dict(collections.Counter(h.get("H1_by") or h.get("H1_human") for h in H)),
    pangram_max=max(h["pangram_ai_frac"] for h in H if h.get("pangram_ai_frac") is not None),
    review_scores=sum(1 for h in H if h.get("iclr_rating") is not None),
    review_score_range=[min(h["iclr_rating"] for h in H if h.get("iclr_rating") is not None), st.median([h["iclr_rating"] for h in H if h.get("iclr_rating") is not None]), max(h["iclr_rating"] for h in H if h.get("iclr_rating") is not None)],
    distinct_anchors=len({h["arxiv"] for h in H}),
    human_type=dict(collections.Counter(h["paper_type"] for h in H)),
    topic=dict(collections.Counter(TOPIC[p["code"]] for p in B).most_common()),
    topic_by_tier={},
)
for h in H: comp["tier_by_pool"][f"{h.get('pool','cited')}|{h['sim_tier']}"] += 1
tb = collections.defaultdict(collections.Counter)
for p in B: tb[TOPIC[p["code"]]][p["human_primary"]["sim_tier"]] += 1
comp["topic_by_tier"] = {k: dict(v) for k, v in tb.items()}
out["composition"] = comp

# ---- balance and trivial features ----
def auroc(pos, neg):
    n = 0; s_ = 0.0
    for a in pos:
        for b in neg:
            n += 1; s_ += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s_ / n
def q(xs): xs = sorted(xs); return [xs[len(xs) // 4], st.median(xs), xs[3 * len(xs) // 4]]
bal = {}
for feat in ["body_words", "appendix_words", "n_sections", "n_subsections", "n_figures", "n_tables", "n_equations", "n_algorithms", "n_unique_cites", "n_internal_refs"]:
    ok = [(a.get(feat), h.get(feat)) for a, h in zip(A, H) if a.get(feat) is not None and h.get(feat) is not None]
    ai = [x for x, _ in ok]; hu = [y for _, y in ok]
    au = auroc(ai, hu)
    bal[feat] = dict(n=len(ok), ai_q=q(ai), hu_q=q(hu), auroc_ai_pos=round(au, 3), separability=round(max(au, 1 - au), 3),
                     pairs_human_larger=sum(1 for a, h in ok if h > a), ties=sum(1 for a, h in ok if h == a), pairs_ai_larger=sum(1 for a, h in ok if a > h))
ratio = [h["body_words"] / a["body_words"] for a, h in zip(A, H)]
bal["body_ratio"] = dict(min=round(min(ratio), 2), q=[round(x, 2) for x in q(ratio)], max=round(max(ratio), 2), n_gt_2_5=sum(1 for x in ratio if x > 2.5), n_lt_1=sum(1 for x in ratio if x < 1))
bal["ai_with_appendix"] = sum(1 for a in A if a.get("appendix_words"))
bal["hu_with_appendix"] = sum(1 for h in H if h.get("appendix_words"))
bal["hu_code_link"] = sum(1 for h in H if h.get("github"))
bal["hu_method_fig"] = sum(1 for h in H if h.get("has_method_fig"))
bal["hu_related_work_section"] = sum(1 for h in H if h.get("has_related_work"))
bal["ai_related_work_section"] = sum(1 for a in A if a.get("has_related_work"))
def tmpl(t):
    t = t if isinstance(t, str) else (",".join(t) if t else "")
    for k in ["iclr", "neurips", "icml", "acl", "colm", "cvpr", "emnlp"]:
        if k in t.lower(): return k.upper() if k != "neurips" else "NeurIPS"
    return "other/none"
bal["hu_template"] = dict(collections.Counter(tmpl(h.get("template")) for h in H).most_common())
bal["ai_template"] = dict(collections.Counter(str(a.get("template")) for a in A).most_common(2))
out["balance"] = bal

# ---- alignment audit (copied) ----
out["alignment"] = {k: ALIGN[k] for k in ALIGN if k not in ("diag", "rank_a", "rank_f", "codes")}

json.dump(out, open(f"{SB}/bench165/results/bench_stats_0912.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
