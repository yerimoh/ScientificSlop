r"""Year by year, does a highly rated paper avoid slop, and does a poorly rated one carry it.

The review scale changes from year to year (2020 uses {1,3,6,8}, 2023 {3,5,6,8,10}, 2017 a 1-10 scale), so a raw
score is not comparable across papers and neither is a percentile taken over the pooled corpus. Everything here is
computed inside one year. For each year and system we split that year's papers into the top and bottom third of the
public review score, and report the share of each third that lands in the upper half of the same year's distribution
of that system's AI-likeness. With no association both shares are 50 percent.

Score definitions compared (the "high and low" the user asked to adjust):
  tercile   top and bottom third of the year's mean review score
  quartile  top and bottom quarter
  decision  orals and spotlights against rejected papers
  absolute  mean rating >= 7 against <= 4, on the years whose scale allows it
Outputs results/year_rate.json, printed in full.
"""
import json, os, sys, collections
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import load, ORDER  # noqa: E402
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P, S, DIR, LAB = load(ours="agg3")
LABEL = LAB
KEYS = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3", "agg5", "macro_redund", "xsec_ref", "citation", "evidence_gap", "argument_graph"]
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v.get("rating") is not None}
years = sorted({v["year"] for v in iclr.values()})


def splits(vs):
    r = np.array([v["rating"] for v in vs], float)
    q33, q67 = np.percentile(r, [33.333, 66.667]); q25, q75 = np.percentile(r, [25, 75])
    out = {"tercile": ([v for v in vs if v["rating"] <= q33], [v for v in vs if v["rating"] >= q67]),
           "quartile": ([v for v in vs if v["rating"] <= q25], [v for v in vs if v["rating"] >= q75]),
           "decision": ([v for v in vs if v["group"] == "reject"], [v for v in vs if v["group"] == "oral"]),
           "absolute": ([v for v in vs if v["rating"] <= 4], [v for v in vs if v["rating"] >= 7])}
    return out


OUT = {"note": __doc__, "years": {}, "summary": {}}
for y in years:
    vs = [v for v in iclr.values() if v["year"] == y]
    OUT["years"][y] = {"n": len(vs), "splits": {}}
    sp = splits(vs)
    for defn, (low, high) in sp.items():
        cell = {"n_low": len(low), "n_high": len(high),
                "mean_rating_low": round(float(np.mean([v["rating"] for v in low])), 2) if low else None,
                "mean_rating_high": round(float(np.mean([v["rating"] for v in high])), 2) if high else None, "systems": {}}
        for k in KEYS:
            have = [v for v in vs if v["id"] in S.get(k, {})]
            if len(have) < 20: continue
            vals = np.array([S[k][v["id"]] * DIR[k] for v in have], float)
            med = float(np.median(vals))
            # a binary item (evidence gap is 0 or 1) has no interior median, so its carriers are the papers at the top value
            binary = len(set(vals)) <= 2 or med in (float(vals.min()), float(vals.max()))
            top = float(vals.max())
            def share(g):
                g = [v for v in g if v["id"] in S[k]]
                if len(g) < 8: return None, 0
                hit = (lambda x: x >= top) if binary else (lambda x: x > med)
                return round(float(np.mean([1.0 if hit(S[k][v["id"]] * DIR[k]) else 0.0 for v in g])) * 100, 1), len(g)
            sl, nl = share(low); sh, nh = share(high)
            if sl is None or sh is None: continue
            cell["systems"][k] = {"share_low": sl, "share_high": sh, "gap": round(sh - sl, 1), "n_low": nl, "n_high": nh}
        OUT["years"][y]["splits"][defn] = cell
for defn in ["tercile", "quartile", "decision", "absolute"]:
    OUT["summary"][defn] = {}
    for k in KEYS:
        gaps = [(y, OUT["years"][y]["splits"][defn]["systems"][k]["gap"]) for y in years
                if k in OUT["years"][y]["splits"][defn]["systems"]]
        if len(gaps) < 4: continue
        g = [x for _, x in gaps]
        neg = sum(1 for x in g if x < 0)
        trend = stats.spearmanr([y for y, _ in gaps], g)
        OUT["summary"][defn][k] = {"years": len(g), "mean_gap": round(float(np.mean(g)), 1), "median_gap": round(float(np.median(g)), 1),
                                   "negative_years": f"{neg}/{len(g)}", "sign_test_p": float(f"{stats.binomtest(neg, len(g), 0.5).pvalue:.3g}"),
                                   "per_year": {str(y): x for y, x in gaps},
                                   "trend_rho": round(float(trend[0]), 3), "trend_p": float(f"{trend[1]:.3g}")}
json.dump(OUT, open(f"{HERE}/results/year_rate.json", "w"), indent=1)

print("share of a year's papers that land in the upper half of that year's AI-likeness, low vs high review score")
for defn in ["tercile", "quartile", "decision", "absolute"]:
    print(f"\n=== split = {defn}")
    print(f"  {'system':22s} {'mean gap':>9s} {'median':>8s} {'years<0':>9s} {'sign p':>9s} {'trend rho':>10s}")
    for k, v in OUT["summary"][defn].items():
        print(f"  {LAB.get(k,k):22s} {v['mean_gap']:+9.1f} {v['median_gap']:+8.1f} {v['negative_years']:>9s} {v['sign_test_p']:>9.3g} {v['trend_rho']:+10.3f}")
print("\nper year gap, tercile split (negative = highly rated papers carry less of it)")
hdr = [k for k in KEYS if k in OUT["summary"]["tercile"]]
print(f"  {'year':6s} " + " ".join(f"{LAB.get(k,k)[:11]:>12s}" for k in hdr))
for y in years:
    c = OUT["years"][y]["splits"]["tercile"]
    print(f"  {y:<6d} " + " ".join(f"{c['systems'][k]['gap']:+12.1f}" if k in c["systems"] else f"{'-':>12s}" for k in hdr))
print("\n  rating of the two thirds per year")
for y in years:
    c = OUT["years"][y]["splits"]["tercile"]
    print(f"  {y} n={OUT['years'][y]['n']:4d} low {c['n_low']:3d} papers mean {c['mean_rating_low']}, high {c['n_high']:3d} papers mean {c['mean_rating_high']}")
