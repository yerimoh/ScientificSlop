"""Alternative data compositions for panel (a) of Fig 3 (0916 night, user: the composition does not support the claim).
Three compositions, every system, same y (within-system percentile, AI-oriented):
  Q   within-year rating quartiles (Q1 low .. Q4 high). Each year contributes equally to every bin by construction,
      and the year-specific rating scales (ICLR 2020 used {1,3,6,8}) no longer mix inside a bin.
  E   same era as FARS, ICLR 2024-2025 only, absolute bands 2-3/4-5/6-7/8-9 (one rating scale, no drift across years).
  H   hard balanced subsample, equal n per year per absolute band (min cell of the year, years with min >= 5).
Writes results/compositions.json and prints a comparison table."""
import json, os, sys, collections
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import load, HERE, BANDS, ORDER
rng = np.random.default_rng(0)
P, S, DIR, LABEL = load("agg5")
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v.get("rating") is not None and v["year"] <= 2025}
# within-year quartile (common split)
Q = {}
for y in range(2017, 2026):
    ids = [k for k, v in iclr.items() if v["year"] == y]; r = np.array([iclr[k]["rating"] for k in ids])
    cuts = np.percentile(r, [25, 50, 75])
    for k in ids: Q[k] = int(np.searchsorted(cuts, iclr[k]["rating"], side="right"))
def pct(d, ids, sign):
    vals = np.array([d[i] * sign for i in ids], float); return dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))
def ci(v, n=2000):
    v = np.asarray(v, float); b = v[rng.integers(0, len(v), (n, len(v)))].mean(1); return [round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]
def run(ids_all, bin_of, nbins):
    res = {}
    for name in ORDER + ["agg3", "agg4"]:
        ids = [i for i in ids_all if i in S[name]]
        if len(ids) < 40: continue
        pl = pct(S[name], ids, DIR[name])
        bins = [[pl[i] for i in ids if bin_of(i) == k] for k in range(nbins)]
        rho, p = stats.spearmanr([iclr[i]["rating"] if bin_of is not Q.get else Q[i] for i in ids], [pl[i] for i in ids])
        rho_raw, p_raw = stats.spearmanr([iclr[i]["rating_pct_in_year"] for i in ids], [pl[i] for i in ids])
        res[name] = {"n": len(ids), "n_bins": [len(b) for b in bins], "means": [round(float(np.mean(b)), 1) for b in bins], "ci": [ci(b) for b in bins],
                     "rho_vs_within_year_pct": round(float(rho_raw), 3), "p": float(f"{p_raw:.2g}")}
    return res
out = {"note": __doc__, "compositions": {}}
ids_all = list(iclr)
out["compositions"]["Q_within_year_quartile"] = {"bins": ["Q1 (low)", "Q2", "Q3", "Q4 (high)"], "n": len(ids_all),
    "bin_rating_range": [[round(float(min(iclr[i]["rating"] for i in ids_all if Q[i] == k)), 1), round(float(max(iclr[i]["rating"] for i in ids_all if Q[i] == k)), 1)] for k in range(4)],
    "bin_mean_rating": [round(float(np.mean([iclr[i]["rating"] for i in ids_all if Q[i] == k])), 2) for k in range(4)],
    "systems": run(ids_all, lambda i: Q[i], 4)}
era = [i for i in ids_all if iclr[i]["year"] >= 2024 and iclr[i].get("band") is not None]
out["compositions"]["E_2024_2025_bands"] = {"bins": [f"{a}-{b}" for a, b in BANDS], "n": len(era), "systems": run(era, lambda i: iclr[i]["band"], 4)}
sub = []
for y in range(2017, 2026):
    cells = {k: [i for i in ids_all if iclr[i]["year"] == y and iclr[i].get("band") == k] for k in range(4)}
    m = min(len(c) for c in cells.values())
    if m >= 5:
        for c in cells.values(): sub += list(rng.choice(c, m, replace=False))
out["compositions"]["H_equal_year_band"] = {"bins": [f"{a}-{b}" for a, b in BANDS], "n": len(sub), "systems": run(sub, lambda i: iclr[i]["band"], 4)}
json.dump(out, open(f"{HERE}/results/compositions.json", "w"), indent=1)
for cname, c in out["compositions"].items():
    print(f"\n== {cname}  n={c['n']}  bins={c['bins']}", c.get("bin_mean_rating", ""))
    for name, r in c["systems"].items():
        print(f"  {name:11s} n={r['n']:4d} {r['n_bins']} means={r['means']} rho={r['rho_vs_within_year_pct']:+.3f} p={r['p']}")
