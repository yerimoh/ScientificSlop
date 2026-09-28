r"""Further analyses the review-score data allows (0917).

N1 trend robustness  the year gap grows; does it survive an equal-sized sample per year
N2 two axes          per item, separating AI papers against following the reviewers
N3 layer structure   inside the papers Pangram calls fully human, the review score by slop tercile
N4 beyond the rating does slop separate accepted from rejected papers at a matched review score
N5 length and area   slop against paper length and research area, the two obvious confounds
Outputs results/analysis_extra.json
"""
import json, os, sys, collections, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import load  # noqa: E402
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
P, S, DIR, LAB = load(ours="agg3")
YR = json.load(open(f"{HERE}/results/year_rate.json"))
P26 = [json.loads(l) for l in open(f"{HERE}/results/papers_2026.jsonl")]
OUT = {"note": __doc__}

# ---------------------------------------------------------------- N1 trend robustness
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v.get("rating") is not None}
years = sorted({v["year"] for v in iclr.values() if sum(1 for u in iclr.values() if u["year"] == v["year"]) >= 60})


def gap_for(year, key, papers=None, frac=1 / 3):
    vs = papers if papers is not None else [v for v in iclr.values() if v["year"] == year and v["id"] in S[key]]
    if len(vs) < 30: return None
    r = np.array([v["rating"] for v in vs], float)
    lo, hi = np.percentile(r, [100 * frac, 100 * (1 - frac)])
    low = [v for v in vs if v["rating"] <= lo]; high = [v for v in vs if v["rating"] >= hi]
    if len(low) < 8 or len(high) < 8: return None
    med = float(np.median([S[key][v["id"]] * DIR[key] for v in vs]))
    sh = lambda g: 100 * float(np.mean([1.0 if S[key][v["id"]] * DIR[key] > med else 0.0 for v in g]))
    return sh(high) - sh(low)


OUT["trend"] = {}
n_min = min(sum(1 for v in iclr.values() if v["year"] == y and v["id"] in S["agg3"]) for y in years)
for key in ["agg3", "binoculars", "nts", "detectgpt", "citation", "xsec_ref", "macro_redund"]:
    full = [(y, gap_for(y, key)) for y in years]
    full = [(y, g) for y, g in full if g is not None]
    eq = {}
    for y in years:
        pool = [v for v in iclr.values() if v["year"] == y and v["id"] in S[key]]
        if len(pool) < n_min: continue
        gs = [gap_for(y, key, [pool[i] for i in rng.choice(len(pool), n_min, replace=False)]) for _ in range(200)]
        gs = [g for g in gs if g is not None]
        if gs: eq[y] = round(float(np.mean(gs)), 1)
    t_full = stats.spearmanr([y for y, _ in full], [g for _, g in full]) if len(full) >= 5 else (None, None)
    t_eq = stats.spearmanr(list(eq), list(eq.values())) if len(eq) >= 5 else (None, None)
    OUT["trend"][key] = {"per_year": {str(y): round(g, 1) for y, g in full}, "trend_rho": round(float(t_full[0]), 3),
                         "trend_p": float(f"{t_full[1]:.3g}"), "equal_n_per_year": int(n_min), "per_year_equal_n": eq,
                         "trend_rho_equal_n": round(float(t_eq[0]), 3) if t_eq[0] is not None else None,
                         "trend_p_equal_n": float(f"{t_eq[1]:.3g}") if t_eq[1] is not None else None,
                         "first3_mean": round(float(np.mean([g for y, g in full if y <= 2019])), 1),
                         "last3_mean": round(float(np.mean([g for y, g in full if y >= 2023])), 1)}


# ---------------------------------------------------------------- N2 two axes per item
def auroc(pos, neg):
    return round(float(sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))), 3)


B = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/bench165"
OUT["two_axes"] = {}
for it in ["macro_redund", "xsec_ref", "citation", "evidence_gap", "argument_graph", "agg3"]:
    prov = None
    f = f"{B}/results/slop/{it}/papers.jsonl"
    if os.path.exists(f):
        rr = [json.loads(l) for l in open(f)]
        ai = [r["slop_score"] for r in rr if r["corpus"] == "AI" and r.get("slop_score") is not None]
        hu = [r["slop_score"] for r in rr if r["corpus"] == "HU" and r.get("slop_score") is not None]
        if ai and hu: prov = auroc(ai, hu)
    if it == "agg3":
        rows = {}
        for sub in ("macro_redund", "xsec_ref", "citation"):
            for l in open(f"{B}/results/slop/{sub}/papers.jsonl"):
                r = json.loads(l)
                rows.setdefault(r["id"], {})[sub] = (r.get("slop_score_agg") if sub == "macro_redund" else r.get("slop_score"))
                rows[r["id"]]["corpus"] = r["corpus"]
        vals = {k: ((v["macro_redund"] + v["xsec_ref"]) / 2 + v["citation"]) / 2 for k, v in rows.items()
                if None not in (v.get("macro_redund"), v.get("xsec_ref"), v.get("citation"))}
        ai = [x for k, x in vals.items() if rows[k]["corpus"] == "AI"]; hu = [x for k, x in vals.items() if rows[k]["corpus"] == "HU"]
        prov = auroc(ai, hu)
    g = YR["summary"]["tercile"].get(it)
    q26 = None
    if it in ("macro_redund", "xsec_ref", "citation", "agg3"):
        xy = [(v[it], v["rating"]) for v in P26 if v.get(it) is not None]
        if len(xy) >= 100: q26 = round(float(stats.spearmanr([a for a, _ in xy], [b for _, b in xy])[0]), 3)
    OUT["two_axes"][it] = {"provenance_auroc": prov, "quality_gap_pp": (g or {}).get("mean_gap"),
                           "quality_years_negative": (g or {}).get("negative_years"), "quality_rho_2026": q26}

# ---------------------------------------------------------------- N3 layer structure
clean = [v for v in P26 if v.get("fraction_ai") == 0 and v.get("agg3") is not None]
allp = [v for v in P26 if v.get("agg3") is not None]


def by_tercile(sel, key="agg3"):
    if len(sel) < 40: return None
    q = np.percentile([v[key] for v in sel], [33.333, 66.667])
    g = {"low slop": [v for v in sel if v[key] <= q[0]], "middle": [v for v in sel if q[0] < v[key] < q[1]],
         "high slop": [v for v in sel if v[key] >= q[1]]}
    out = {}
    for name, vs in g.items():
        out[name] = {"n": len(vs), "mean_rating": round(float(np.mean([v["rating"] for v in vs])), 2),
                     "accept_rate": round(100 * float(np.mean([1.0 if v["accept"] else 0.0 for v in vs])), 1)}
    a, b = [v["rating"] for v in g["low slop"]], [v["rating"] for v in g["high slop"]]
    out["low_vs_high_p"] = float(f"{stats.mannwhitneyu(a, b).pvalue:.3g}")
    out["rating_difference"] = round(float(np.mean(a) - np.mean(b)), 2)
    return out


OUT["layer"] = {"all 2026 papers": by_tercile(allp), "Pangram calls fully human": by_tercile(clean),
                "n_all": len(allp), "n_clean": len(clean)}

# ---------------------------------------------------------------- N4 decision at matched rating
sel = [v for v in P26 if v.get("agg3") is not None]
bins = collections.defaultdict(list)
for v in sel: bins[round(v["rating"] * 2) / 2].append(v)
rows = []
for b, vs in sorted(bins.items()):
    acc = [v["agg3"] for v in vs if v["accept"]]; rej = [v["agg3"] for v in vs if not v["accept"]]
    if len(acc) >= 10 and len(rej) >= 10:
        rows.append({"rating": b, "n_accept": len(acc), "n_reject": len(rej), "auroc_reject_over_accept": auroc(rej, acc)})
OUT["decision_at_matched_rating"] = {"bins": rows, "pooled_auroc": round(float(np.mean([r["auroc_reject_over_accept"] for r in rows])), 3) if rows else None}

# ---------------------------------------------------------------- N5 length and area
OUT["confounds_2026"] = {}
for k in ["agg3", "citation", "xsec_ref", "macro_redund", "pangram"]:
    xy = [(v[k], v["words"]) for v in P26 if v.get(k) is not None and v.get("words")]
    if len(xy) >= 100:
        OUT["confounds_2026"][k] = {"rho_vs_length": round(float(stats.spearmanr([a for a, _ in xy], [b for _, b in xy])[0]), 3)}
areas = collections.Counter(v.get("area") for v in P26 if v.get("agg3") is not None)
big = [a for a, n in areas.items() if n >= 30 and a]
OUT["by_area"] = {}
for a in big:
    vs = [v for v in P26 if v.get("area") == a and v.get("agg3") is not None]
    OUT["by_area"][a] = {"n": len(vs), "mean_agg3": round(float(np.mean([v["agg3"] for v in vs])), 3),
                         "rho_rating": round(float(stats.spearmanr([v["agg3"] for v in vs], [v["rating"] for v in vs])[0]), 3)}
json.dump(OUT, open(f"{HERE}/results/analysis_extra.json", "w"), indent=1)

print("N1 the year gap over time (pp, negative = highly rated papers carry less)")
for k, v in OUT["trend"].items():
    print(f"  {LAB.get(k,k):22s} 2017-19 {v['first3_mean']:+6.1f} -> 2023-25 {v['last3_mean']:+6.1f}   rho={v['trend_rho']:+.3f} (p={v['trend_p']:.3g})   equal n={v['equal_n_per_year']} rho={v['trend_rho_equal_n']}")
print("\nN2 two axes per item")
print(f"  {'item':22s} {'AUROC':>7s} {'gap pp':>8s} {'years<0':>9s} {'rho 2026':>9s}")
for k, v in OUT["two_axes"].items():
    print(f"  {LAB.get(k,k):22s} {str(v['provenance_auroc']):>7s} {str(v['quality_gap_pp']):>8s} {str(v['quality_years_negative']):>9s} {str(v['quality_rho_2026']):>9s}")
print("\nN3 review score by slop tercile, ICLR 2026")
for tag in ["all 2026 papers", "Pangram calls fully human"]:
    d = OUT["layer"][tag]
    if not d: continue
    print(f"  {tag}")
    for name in ["low slop", "middle", "high slop"]:
        print(f"    {name:11s} n={d[name]['n']:4d} mean rating {d[name]['mean_rating']:.2f}  accept {d[name]['accept_rate']:.0f}%")
    print(f"    low minus high {d['rating_difference']:+.2f} rating points, p={d['low_vs_high_p']:.3g}")
print("\nN4 rejected over accepted at a matched review score")
for r in OUT["decision_at_matched_rating"]["bins"]:
    print(f"  rating {r['rating']:<4} n {r['n_reject']:3d}/{r['n_accept']:3d}  auroc {r['auroc_reject_over_accept']:.3f}")
print(f"  pooled {OUT['decision_at_matched_rating']['pooled_auroc']}")
print("\nN5 confounds, ICLR 2026")
for k, v in OUT["confounds_2026"].items(): print(f"  {LAB.get(k,k):22s} rho vs length {v['rho_vs_length']:+.3f}")
for a, v in OUT["by_area"].items(): print(f"  area {a[:34]:34s} n={v['n']:3d} mean {v['mean_agg3']:.3f} rho {v['rho_rating']:+.3f}")
