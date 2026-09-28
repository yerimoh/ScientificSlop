"""Paragraph 3 of §5.2 (ANALYSIS_SECTION_PLAN_0914 §3): slop and review scores.

(a) review_corr.json : Spearman between each system's score and the public ICLR rating on the
    40 rated human anchors of SciSlopBench (same sample as bench165 m5_vs_true_rating), extended
    with argument_graph, evidence_gap and the SciSlop aggregates. Figure--text (fig_graph) is
    excluded on purpose (user decision 0914).
(b) stairs.json      : mean slop per group FARS / ICLR reject / ICLR accept / ICLR oral for the four
    deterministic items and their aggregate, with bootstrap CIs, trend tests, year strata, and the
    within-ICLR rating correlation (n ~ 1.1k).
REPORT.md is written by report.py from these two files.
"""
import json, os, glob, math, random, collections
import numpy as np
from scipy import stats

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{HERE}/results"
random.seed(0); np.random.seed(0)

ITEMS4 = ["macro_redund", "xsec_ref", "citation", "evidence_gap"]
# 0916: the two model-based items, present when their ICLR runs exist (review/scripts/{run_slop_iclr --checker argument_graph, figexp_score}.py)
EXTRA_ITEMS = [it for it in ["argument_graph", "fig_exposition"] if os.path.exists(f"{HERE}/results/slop/{it}/papers.jsonl")]
GROUPS4 = ["FARS", "reject", "accept", "oral"]


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def item_score(r, item):
    """Score used for plane/paper aggregation (SLOP_SCORE.md): macro uses the ceiling-rescaled score.
    fig_exposition uses the five pattern kinds (the hand-verified sixth kind does not exist for ICLR)."""
    if item == "macro_redund":
        return r.get("slop_score_agg")
    if item == "fig_exposition":
        return r.get("slop_score_pattern5")
    return r.get("slop_score")


def mean_skip(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def boot_ci(v, n=2000, stat=np.mean):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    idx = np.random.randint(0, len(v), (n, len(v)))
    s = stat(v[idx], axis=1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def spear(x, y, boot=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    idx = np.random.randint(0, len(x), (boot, len(x)))
    bs = []
    for i in idx:
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if bs else [None, None]   # all-constant resamples (a near-zero item within a year)
    return {"n": int(len(x)), "spearman": (round(float(rho), 3) if not np.isnan(rho) else None), "p": (float(f"{p:.3g}") if not np.isnan(p) else None), "ci95": ci}


def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    more = sum((x > b).sum() for x in a); less = sum((x < b).sum() for x in a)
    return round(float((more - less) / (len(a) * len(b))), 4)


# ------------------------------------------------------------------ bench165 scores (AI + HU, 286 items)
items = json.load(open(f"{B}/items165.json"))["items"]
meta = {i["item_id"]: i for i in items}
S = {i["item_id"]: {} for i in items}
for it in items:
    S[it["item_id"]]["triv_body_words"] = it["body_words"]
for name, path, key in [("binoculars", "binoculars.jsonl", "binoculars"), ("detectgpt", "detectgpt.jsonl", "detectgpt"),
                        ("fast_detectgpt", "fast_detectgpt.jsonl", "fast_detectgpt"), ("nts", "nts.jsonl", "nts")]:
    for r in rows(f"{B}/results/{path}"):
        if r.get("id") in S and isinstance(r.get(key), (int, float)):
            S[r["id"]][name] = r[key]
for sysname in ["b3a", "b2h", "b3i"]:
    for f in glob.glob(f"{B}/results/reviews/{sysname}/*.json"):
        r = json.load(open(f)); fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if r.get("item_id") in S and isinstance(v, (int, float)):
            S[r["item_id"]][f"rev_{sysname}"] = v


def slop_id(rid):
    return f"AI_{rid}" if rid.startswith("FA") else f"HU_{rid}"


bench_item_rows = {}
for name in ["macro_redund", "xsec_ref", "argument_graph", "citation", "evidence_gap"]:
    bench_item_rows[name] = rows(f"{B}/results/slop/{name}/papers.jsonl")
# fig_exposition on the pairs: the item's own rows (6 kinds) for (a); the 5-pattern-kind rescoring for (b)
FE = f"{ROOT}/paper/draft_v6/slop/Artifacts/fig_exposition/results/papers.jsonl"
bench_item_rows["fig_exposition"] = [r for r in rows(FE) if r.get("applicable")]
bench_item_rows["fig_exposition_p5"] = rows(f"{R}/slop/fig_exposition/bench_pattern5.jsonl")
for name in ["fig_exposition"]:
    for r in bench_item_rows[name]:
        rid = slop_id(r["id"])
        if rid in S and r.get("slop_score") is not None:
            S[rid][f"slop_{name}"] = r["slop_score"]
    for r in bench_item_rows[name]:
        rid = slop_id(r["id"])
        if rid in S and r.get("slop_score") is not None:
            S[rid][f"slop_{name}"] = r["slop_score"]
            if name == "macro_redund":
                S[rid]["slop_macro_redund_agg"] = r["slop_score_agg"]
# SciSlop aggregates (SLOP_SCORE.md: item -> plane -> paper, unweighted, N/A skipped). fig_graph excluded (0914).
for rid in S:
    s = S[rid]
    structure = mean_skip([s.get("slop_macro_redund_agg"), s.get("slop_xsec_ref")])
    argument5 = mean_skip([s.get("slop_argument_graph"), s.get("slop_citation")])
    argument4 = s.get("slop_citation")
    artifacts = s.get("slop_evidence_gap")
    s["scislop5"] = mean_skip([structure, argument5, artifacts])   # 5 items = all but Figure--text
    s["scislop_det4"] = mean_skip([structure, argument4, artifacts])  # deterministic 4 = what runs on ICLR
    s["plane_structure"] = structure

# ------------------------------------------------------------------ (a) 40 rated anchors
hu_rated = [i for i in S if meta[i]["label"] == 0 and isinstance(meta[i].get("iclr_rating"), (int, float))]
COLS = ["binoculars", "detectgpt", "fast_detectgpt", "nts", "rev_b3a", "rev_b2h", "rev_b3i",
        "slop_macro_redund", "slop_xsec_ref", "slop_argument_graph", "slop_citation", "slop_evidence_gap", "slop_fig_exposition",
        "plane_structure", "scislop_det4", "scislop5", "triv_body_words"]
review_corr = {"sample": "SciSlopBench human papers with a public ICLR rating (items165 iclr_rating)",
               "n_rated": len(hu_rated), "excluded": "fig_graph (Figure--text), by decision 0914", "systems": {}}
for col in COLS:
    xs = [(S[i][col], meta[i]["iclr_rating"]) for i in hu_rated if S[i].get(col) is not None]
    if len(xs) >= 10:
        d = spear([a for a, _ in xs], [b for _, b in xs])
        d["direction_note"] = "slop items and aggregates: higher = more slop, so a negative rho means less slop with higher rating"
        review_corr["systems"][col] = d
# reproduce the archived numbers as a check
old = json.load(open(f"{B}/results/pairwise165.json"))["m5_vs_true_rating"]
review_corr["check_vs_pairwise165"] = {k: {"archived": old[k]["spearman"], "now": review_corr["systems"].get(k, {}).get("spearman")}
                                       for k in old if k in review_corr["systems"]}
json.dump(review_corr, open(f"{R}/review_corr.json", "w"), indent=1)

# ------------------------------------------------------------------ (b) staircase
recs = json.load(open(f"{HERE}/records/iclr_records.json"))["records"]
rec_by_id = {r["id"]: r for r in recs}
ai_codes = {i["pair"] for i in items if i["label"] == 1}
hu_ids = {i["arxiv"] for i in items if i["label"] == 0}

# per-paper table: id, group, year, rating, scores of the 4 items
P = collections.OrderedDict()
for name in ITEMS4:
    for r in bench_item_rows[name]:
        if r["corpus"] == "AI" and r["id"] in ai_codes:
            P.setdefault(r["id"], {"group": "FARS", "year": None, "rating": None, "src": "bench_AI"})[name] = item_score(r, name)
            if name == "macro_redund":
                P[r["id"]]["macro_raw"] = r["slop_score"]; P[r["id"]]["n_words"] = r.get("n_words")
        elif r["corpus"] == "HU" and r["id"] in hu_ids:
            P.setdefault(f"HU_{r['id']}", {"group": "bench_HU", "year": None, "rating": None, "src": "bench_HU"})[name] = item_score(r, name)
            if name == "macro_redund":
                P[f"HU_{r['id']}"]["macro_raw"] = r["slop_score"]; P[f"HU_{r['id']}"]["n_words"] = r.get("n_words")
    for r in rows(f"{R}/slop/{name}/papers.jsonl"):
        m = rec_by_id[r["id"]]
        P.setdefault(r["id"], {"group": m["group4"], "group5": m["group5"], "year": m["year"], "rating": m["overall_mean"], "src": "iclr"})[name] = item_score(r, name)
        if name == "macro_redund":
            P[r["id"]]["macro_raw"] = r["slop_score"]; P[r["id"]]["n_words"] = r.get("n_words")
            P[r["id"]]["source_incomplete"] = r.get("source_incomplete")
# 0916: argument_graph (bench rows for FARS / bench_HU, review run for ICLR) and fig_exposition (5 pattern kinds everywhere)
for name in EXTRA_ITEMS:
    brows = bench_item_rows["fig_exposition_p5"] if name == "fig_exposition" else bench_item_rows[name]
    for r in brows:
        if r["corpus"] == "AI" and r["id"] in ai_codes and r["id"] in P:
            P[r["id"]][name] = item_score(r, name)
        elif r["corpus"] == "HU" and r["id"] in hu_ids and f"HU_{r['id']}" in P:
            P[f"HU_{r['id']}"][name] = item_score(r, name)
    for r in rows(f"{R}/slop/{name}/papers.jsonl"):
        if r["id"] in P and r.get("status", "ok") == "ok":
            P[r["id"]][name] = item_score(r, name)
# quality filter (0915): ICLR papers whose tex extraction failed (records/exclude_quality.json, outcome-independent rules)
EXQ = set(json.load(open(f"{HERE}/records/exclude_quality.json"))["ids"]) if os.path.exists(f"{HERE}/records/exclude_quality.json") else set()
for pid in list(P):
    if pid in EXQ:
        del P[pid]
for pid, p in P.items():
    p["scislop_det4"] = mean_skip([mean_skip([p.get("macro_redund"), p.get("xsec_ref")]), p.get("citation"), p.get("evidence_gap")])
    if "argument_graph" in EXTRA_ITEMS:   # Argument plane = mean(argument_graph, citation); Artifacts stays evidence_gap (fig_exposition is observational)
        p["scislop5"] = mean_skip([mean_skip([p.get("macro_redund"), p.get("xsec_ref")]), mean_skip([p.get("argument_graph"), p.get("citation")]), p.get("evidence_gap")])
LINES = ITEMS4 + EXTRA_ITEMS + ["scislop_det4"] + (["scislop5"] if "argument_graph" in EXTRA_ITEMS else [])


def group_stats(sel, key):
    v = [p[key] for p in sel if p.get(key) is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4),
            "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None, "ci95": boot_ci(v)}


def stairs_for(papers, label):
    out = {"label": label, "groups": {}, "trend": {}, "adjacent": {}}
    by_g = {g: [p for p in papers if p["group"] == g] for g in GROUPS4}
    out["n_by_group"] = {g: len(by_g[g]) for g in GROUPS4}
    for key in LINES + ["macro_raw", "n_words"]:
        out["groups"][key] = {g: group_stats(by_g[g], key) for g in GROUPS4}
    ordinal = {g: k for k, g in enumerate(GROUPS4)}
    for key in LINES:
        xs = [(ordinal[p["group"]], p[key]) for p in papers if p["group"] in ordinal and p.get(key) is not None]
        d = spear([a for a, _ in xs], [b for _, b in xs])
        xs_i = [(a, b) for a, b in xs if a >= 1]
        d_i = spear([a for a, _ in xs_i], [b for _, b in xs_i])
        kw = stats.kruskal(*[[b for a, b in xs_i if a == k] for k in (1, 2, 3)])
        out["trend"][key] = {"ordinal_all4": d, "ordinal_iclr3": d_i,
                             "kruskal_iclr3": {"H": round(float(kw.statistic), 2), "p": float(f"{kw.pvalue:.3g}")}}
        adj = {}
        for g1, g2 in zip(GROUPS4[:-1], GROUPS4[1:]):
            a = [p[key] for p in by_g[g1] if p.get(key) is not None]; b = [p[key] for p in by_g[g2] if p.get(key) is not None]
            if a and b:
                mw = stats.mannwhitneyu(a, b, alternative="two-sided")
                adj[f"{g1}>{g2}"] = {"delta_mean": round(float(np.mean(b) - np.mean(a)), 4), "cliff_delta": cliff(a, b),
                                     "mwu_p": float(f"{mw.pvalue:.3g}")}
        out["adjacent"][key] = adj
    return out


all_p = [p for p in P.values() if p["group"] in GROUPS4]
stairs = {"excluded_item": "fig_graph (Figure--text) by decision 0914" + ("" if EXTRA_ITEMS else "; argument_graph / fig_exposition not yet run on ICLR"),
          "extra_items_0916": EXTRA_ITEMS,
          "score_note": "macro_redund plotted as slop_score_agg = min(1, score/0.10) (SLOP_SCORE.md ceiling); macro_raw kept apart",
          "sources": {"FARS": "bench165 AI 143 (existing results/slop/<item>/papers.jsonl)",
                      "ICLR": "artifact-ai2science/Evaluation/ICLR/data 2017-2025 tex papers, decided ones only"},
          "n_iclr_records": len(recs), "n_iclr_undecided_excluded": sum(1 for r in recs if r["group4"] == "undecided"),
          "n_source_incomplete_iclr": sum(1 for p in P.values() if p.get("src") == "iclr" and p.get("source_incomplete")),
          "n_quality_excluded": len(EXQ)}
stairs["main"] = stairs_for(all_p, "all years 2017-2025")
stairs["post2023"] = stairs_for([p for p in all_p if p["group"] == "FARS" or (p["year"] or 0) >= 2024], "FARS + ICLR 2024-2025 only")
stairs["pre2023"] = stairs_for([p for p in all_p if p["group"] == "FARS" or (p["year"] or 9999) <= 2023], "FARS + ICLR 2017-2023 only")
stairs["complete_source_only"] = stairs_for([p for p in all_p if p["group"] == "FARS" or not p.get("source_incomplete")], "ICLR papers with complete tex only")
# five-level breakdown and per-year means
g5 = ["reject", "poster", "spotlight", "oral"]
stairs["group5_means"] = {key: {g: group_stats([p for p in all_p if p.get("group5") == g], key) for g in g5} for key in LINES}
stairs["by_year"] = {}
for y in range(2017, 2026):
    sel = [p for p in all_p if p.get("year") == y]
    stairs["by_year"][y] = {"n": {g: sum(1 for p in sel if p["group"] == g) for g in GROUPS4[1:]},
                            **{key: {g: (group_stats([p for p in sel if p["group"] == g], key).get("mean")) for g in GROUPS4[1:]} for key in LINES}}
# bench HU reference
stairs["bench_HU_reference"] = {key: group_stats([p for p in P.values() if p["group"] == "bench_HU"], key) for key in LINES}
# within-ICLR rating correlation: rating percentile within year (scales differ across years), pooled + per year
iclr = [p for p in P.values() if p["src"] == "iclr" and p["group"] in GROUPS4 and p.get("rating") is not None]
for y in sorted({p["year"] for p in iclr}):      # every year present (a 2026 record joined the corpus 0916)
    sel = [p for p in iclr if p["year"] == y]
    rk = stats.rankdata([p["rating"] for p in sel]) / len(sel)
    for p, r in zip(sel, rk):
        p["rating_pct_in_year"] = float(r)
stairs["rating_corr_iclr"] = {"note": "Spearman of item score vs public mean rating; pooled uses the within-year percentile of the rating", "pooled": {}, "per_year": {}}
for key in LINES:
    xs = [(p[key], p["rating_pct_in_year"]) for p in iclr if p.get(key) is not None and p.get("rating_pct_in_year") is not None]
    stairs["rating_corr_iclr"]["pooled"][key] = spear([a for a, _ in xs], [b for _, b in xs])
    stairs["rating_corr_iclr"]["per_year"][key] = {}
    for y in range(2017, 2026):
        xs = [(p[key], p["rating"]) for p in iclr if p["year"] == y and p.get(key) is not None]
        if len(xs) >= 10:
            stairs["rating_corr_iclr"]["per_year"][key][y] = spear([a for a, _ in xs], [b for _, b in xs], boot=500)
# length guard: within ICLR, Spearman of score vs ordinal group and vs rating after residualising on log(n_words)
def partial_spear(x, y, z):
    x, y, z = (stats.rankdata(np.asarray(v, float)) for v in (x, y, z))
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rho, p = stats.pearsonr(resid(x, z), resid(y, z))
    return {"n": int(len(x)), "partial_spearman": round(float(rho), 3), "p": float(f"{p:.3g}")}
ordinal = {g: k for k, g in enumerate(GROUPS4)}
stairs["length_guard_iclr"] = {"note": "ICLR papers only; control = log body words from macro_redund n_words", "vs_ordinal_group": {}, "vs_rating_pct": {}, "score_vs_words": {}}
for key in LINES:
    sel = [p for p in iclr if p.get(key) is not None and p.get("n_words")]
    lw = [math.log(p["n_words"]) for p in sel]
    stairs["length_guard_iclr"]["vs_ordinal_group"][key] = partial_spear([p[key] for p in sel], [ordinal[p["group"]] for p in sel], lw)
    stairs["length_guard_iclr"]["vs_rating_pct"][key] = partial_spear([p[key] for p in sel], [p["rating_pct_in_year"] for p in sel], lw)
    stairs["length_guard_iclr"]["score_vs_words"][key] = spear([p[key] for p in sel], lw, boot=200)
stairs["length_guard_iclr"]["words_vs_group"] = spear([math.log(p["n_words"]) for p in iclr if p.get("n_words")], [ordinal[p["group"]] for p in iclr if p.get("n_words")], boot=200)
json.dump(stairs, open(f"{R}/stairs.json", "w"), indent=1)
print("\n  length guard (ICLR only, partial Spearman | log words):")
for key in LINES:
    g = stairs["length_guard_iclr"]
    print(f"  {key:14s} vs group rho={g['vs_ordinal_group'][key]['partial_spearman']:+.3f} p={g['vs_ordinal_group'][key]['p']:.2g} | vs rating rho={g['vs_rating_pct'][key]['partial_spearman']:+.3f} p={g['vs_rating_pct'][key]['p']:.2g} | score~words rho={g['score_vs_words'][key]['spearman']:+.3f}")
print("  words~group rho=", stairs["length_guard_iclr"]["words_vs_group"]["spearman"])
with open(f"{R}/papers_stairs.jsonl", "w") as fh:
    for pid, p in P.items():
        fh.write(json.dumps({"id": pid, **p}) + "\n")

# console summary
print("(a) review corr, n =", review_corr["n_rated"])
for k, v in review_corr["systems"].items():
    print(f"  {k:22s} n={v['n']:3d} rho={v['spearman']:+.3f} p={v['p']:.3g} ci={v['ci95']}")
print("check vs archive:", review_corr["check_vs_pairwise165"])
print("\n(b) stairs, n by group:", stairs["main"]["n_by_group"], "| undecided excluded:", stairs["n_iclr_undecided_excluded"],
      "| incomplete tex:", stairs["n_source_incomplete_iclr"])
for key in LINES + ["macro_raw", "n_words"]:
    g = stairs["main"]["groups"][key]
    print(f"  {key:14s} " + " | ".join(f"{gr}: {g[gr].get('mean')} (n={g[gr]['n']})" for gr in GROUPS4))
for key in LINES:
    t = stairs["main"]["trend"][key]
    print(f"  trend {key:14s} all4 rho={t['ordinal_all4']['spearman']:+.3f} p={t['ordinal_all4']['p']:.2g} | iclr3 rho={t['ordinal_iclr3']['spearman']:+.3f} p={t['ordinal_iclr3']['p']:.2g} | KW p={t['kruskal_iclr3']['p']:.2g}")
    print(f"        adjacent: " + " ; ".join(f"{k} d={v['cliff_delta']:+.3f} p={v['mwu_p']:.2g}" for k, v in stairs["main"]["adjacent"][key].items()))
print("\n  post-2023 (ICLR 2024-25) means:")
for key in LINES:
    g = stairs["post2023"]["groups"][key]
    print(f"  {key:14s} " + " | ".join(f"{gr}: {g[gr].get('mean')} (n={g[gr]['n']})" for gr in GROUPS4))
print("\n  rating corr within ICLR (pooled, within-year percentile):")
for key in LINES:
    v = stairs["rating_corr_iclr"]["pooled"][key]
    print(f"  {key:14s} n={v['n']} rho={v['spearman']:+.3f} p={v['p']:.2g} ci={v['ci95']}")
