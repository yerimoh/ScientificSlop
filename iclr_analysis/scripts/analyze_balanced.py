"""Year x domain balanced version of Fig 3 (0916). Question raised by the user: the score bands are not comparable
because the 2-3 band is 45% ICLR 2020 and 27% ICLR 2025 while the 8-9 band spreads over all years, and detector
percentiles drift by year (Binoculars 65 in 2017 -> 34 in 2025), so the band curves mix year drift with rating.
Fix: post-stratification. Every band (panel a) and every ICLR group (panel b, c) is reweighted so that its
year x domain composition equals the pooled composition of the ICLR sample, restricted to strata present in every
band (common support). y stays the within-system percentile of the score, AI-oriented. Correlations are partial
Spearman with the year x domain stratum removed by rank regression, plus the mean of within-year Spearman.
Also writes a hard balanced subsample (equal n per year per band, seed 0) as a check.
Output results/balanced.json; the figure is drawn by fig_review_3panel_balanced.py."""
import json, os, collections, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import load, HERE, GROUPS, BANDS, ORDER, ITEMS
rng = np.random.default_rng(0)
P, S, DIR, LABEL = load("agg5")
iclr_ids = [k for k, v in P.items() if v["src"] == "iclr" and v.get("band") is not None and v["year"] <= 2025]
STRATA = sorted({P[i]["stratum"] for i in iclr_ids})


def pct(d, ids, sign):
    vals = np.array([d[i] * sign for i in ids], float)
    return dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))


def ps_weights(ids, cell_of, target_ids=None):
    """post-stratification weights: cell composition of each level of cell_of -> pooled composition over target_ids,
    restricted to strata present in every level. returns (weights, dropped_ids, n_strata_used)"""
    target_ids = target_ids or ids
    levels = sorted({cell_of(i) for i in ids})
    present = [set(P[i]["stratum"] for i in ids if cell_of(i) == L) for L in levels]
    common = set.intersection(*present) if present else set()
    tgt = collections.Counter(P[i]["stratum"] for i in target_ids if P[i]["stratum"] in common); Nt = sum(tgt.values())
    w, dropped = {}, []
    for L in levels:
        sel = [i for i in ids if cell_of(i) == L]
        cnt = collections.Counter(P[i]["stratum"] for i in sel); n = sum(v for s, v in cnt.items() if s in common)
        for i in sel:
            s = P[i]["stratum"]
            if s not in common: dropped.append(i); continue
            w[i] = (tgt[s] / Nt) / (cnt[s] / n)
    return w, dropped, len(common)


def wmean(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    return float((y * w).sum() / w.sum())


def wmean_ci(y, w, n=2000):
    y, w = np.asarray(y, float), np.asarray(w, float)
    idx = rng.integers(0, len(y), (n, len(y)))
    b = (y[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]


def partial_spear(x, y, strata):
    """Spearman after removing stratum means of the ranks (one-hot rank regression)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    levels = sorted(set(strata)); idx = {s: k for k, s in enumerate(levels)}
    X = np.zeros((len(x), len(levels)))
    for k, s in enumerate(strata): X[k, idx[s]] = 1
    def resid(a): return a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    r, p = stats.pearsonr(resid(rx), resid(ry))
    return {"rho": round(float(r), 3), "p": float(f"{p:.2g}"), "n": int(len(x))}


def within_year_mean_rho(x, y, years, min_n=20):
    per = {}
    for yv in sorted(set(years)):
        sel = [k for k, yy in enumerate(years) if yy == yv]
        if len(sel) >= min_n:
            per[yv] = round(float(stats.spearmanr([x[k] for k in sel], [y[k] for k in sel])[0]), 3)
    return {"per_year": per, "mean": round(float(np.mean(list(per.values()))), 3) if per else None,
            "n_years_negative": sum(1 for v in per.values() if v < 0), "n_years": len(per)}


out = {"note": __doc__, "strata": STRATA, "composition": {}, "panel_a": {}, "panel_b": {}, "panel_c": {}, "subsample": {}}
# composition tables (for the report)
comp = {"year_by_band": {}, "domain_by_band": {}, "year_by_group": {}, "domain_by_group": {}}
for k, (key, lev) in {"year_by_band": ("year", range(len(BANDS))), "domain_by_band": ("domain", range(len(BANDS))),
                      "year_by_group": ("year", ["reject", "accept", "oral"]), "domain_by_group": ("domain", ["reject", "accept", "oral"])}.items():
    c = collections.Counter((P[i][key], P[i]["band"] if "band" in k else P[i]["group"]) for i in iclr_ids)
    rows = sorted({P[i][key] for i in iclr_ids})
    comp[k] = {str(r): {str(L): c[(r, L)] for L in lev} for r in rows}
out["composition"] = comp
# ---- panel (a): bands
for name in ORDER + ["agg3", "agg4", "agg5"]:
    ids = [i for i in S[name] if i in P and P[i]["src"] == "iclr" and P[i].get("band") is not None and P[i]["year"] <= 2025]
    pl = pct(S[name], ids, DIR[name])
    w, dropped, ns = ps_weights(ids, lambda i: P[i]["band"])
    raw = [float(np.mean([pl[i] for i in ids if P[i]["band"] == k])) for k in range(len(BANDS))]
    bal, ci, nb = [], [], []
    for k in range(len(BANDS)):
        sel = [i for i in ids if P[i]["band"] == k and i in w]
        bal.append(round(wmean([pl[i] for i in sel], [w[i] for i in sel]), 2)); ci.append(wmean_ci([pl[i] for i in sel], [w[i] for i in sel])); nb.append(len(sel))
    x = [P[i]["rating"] for i in ids]; y = [pl[i] for i in ids]
    rho, p = stats.spearmanr(x, y)
    out["panel_a"][name] = {"label": LABEL[name], "n": len(ids), "n_used": len(w), "n_dropped_common_support": len(dropped), "n_strata": ns, "n_by_band": nb,
                            "raw_bands": [round(v, 2) for v in raw], "balanced_bands": bal, "balanced_ci": ci,
                            "rho_raw": {"rho": round(float(rho), 3), "p": float(f"{p:.2g}")},
                            "rho_partial_year_domain": partial_spear(x, y, [P[i]["stratum"] for i in ids]),
                            "rho_partial_year": partial_spear(x, y, [P[i]["year"] for i in ids]),
                            "within_year": within_year_mean_rho(x, y, [P[i]["year"] for i in ids])}
# ---- panel (b): groups, FARS + ICLR pooled percentile
for name in ORDER + ["agg3", "agg4", "agg5"]:
    ids = [i for i in S[name] if i in P and (P[i]["src"] != "iclr" or (P[i].get("band") is not None and P[i]["year"] <= 2025))]
    pr = pct(S[name], ids, DIR[name])
    ic = [i for i in ids if P[i]["src"] == "iclr"]
    w, dropped, ns = ps_weights(ic, lambda i: P[i]["group"])
    raw = [float(np.mean([pr[i] for i in ids if P[i]["group"] == g])) for g in GROUPS]
    bal, ci, ng = [], [], []
    for g in GROUPS:
        sel = [i for i in ids if P[i]["group"] == g and (g == "FARS" or i in w)]
        ww = [1.0 if g == "FARS" else w[i] for i in sel]
        bal.append(round(wmean([pr[i] for i in sel], ww), 2)); ci.append(wmean_ci([pr[i] for i in sel], ww)); ng.append(len(sel))
    ordv = [GROUPS.index(P[i]["group"]) for i in ic]; yv = [pr[i] for i in ic]
    out["panel_b"][name] = {"label": LABEL[name], "n_by_group": ng, "n_dropped_common_support": len(dropped), "n_strata": ns,
                            "raw_groups": [round(v, 2) for v in raw], "balanced_groups": bal, "balanced_ci": ci,
                            "rho_iclr_raw": round(float(stats.spearmanr(ordv, yv)[0]), 3),
                            "rho_iclr_partial_year_domain": partial_spear(ordv, yv, [P[i]["stratum"] for i in ic])}
# ---- panel (c): item means per group, balanced within ICLR
for k, lab, *_ in ITEMS:
    ids = [i for i in S[k] if i in P and (P[i]["src"] != "iclr" or (P[i].get("band") is not None and P[i]["year"] <= 2025))]
    ic = [i for i in ids if P[i]["src"] == "iclr"]
    w, dropped, ns = ps_weights(ic, lambda i: P[i]["group"])
    g_out = {}
    for g in GROUPS:
        sel = [i for i in ids if P[i]["group"] == g and (g == "FARS" or i in w)]
        ww = [1.0 if g == "FARS" else w[i] for i in sel]
        vals = [S[k][i] for i in sel]
        g_out[g] = {"n": len(sel), "raw_mean": round(float(np.mean([S[k][i] for i in ids if P[i]["group"] == g])), 4) if any(P[i]["group"] == g for i in ids) else None,
                    "mean": round(wmean(vals, ww), 4), "ci95": [round(v / 1.0, 4) for v in wmean_ci(vals, ww)]}
    ordv = [GROUPS.index(P[i]["group"]) for i in ic]; yv = [S[k][i] for i in ic]
    out["panel_c"][k] = {"label": lab, "groups": g_out, "n_dropped_common_support": len(dropped), "n_strata": ns,
                         "rho_iclr_raw": round(float(stats.spearmanr(ordv, yv)[0]), 3),
                         "rho_iclr_partial_year_domain": partial_spear(ordv, yv, [P[i]["stratum"] for i in ic]),
                         "rho_rating_partial_year_domain": partial_spear([P[i]["rating"] for i in ic], yv, [P[i]["stratum"] for i in ic])}
# ---- hard subsample check: equal n per (year, band), n = min over bands within the year, years with min >= 5
sub_ids = []
for yv in range(2017, 2026):
    cells = {k: [i for i in iclr_ids if P[i]["year"] == yv and P[i]["band"] == k] for k in range(len(BANDS))}
    m = min(len(c) for c in cells.values())
    if m >= 5:
        for k, c in cells.items(): sub_ids += list(rng.choice(c, m, replace=False))
out["subsample"]["rule"] = "equal n per year per band = min cell of the year, years whose min cell >= 5, seed 0"
out["subsample"]["years_used"] = sorted({P[i]["year"] for i in sub_ids}); out["subsample"]["n"] = len(sub_ids)
out["subsample"]["systems"] = {}
for name in ORDER:
    ids = [i for i in sub_ids if i in S[name]]
    if len(ids) < 40: continue
    pl = pct(S[name], ids, DIR[name])
    rho, p = stats.spearmanr([P[i]["rating"] for i in ids], [pl[i] for i in ids])
    out["subsample"]["systems"][name] = {"n": len(ids), "bands": [round(float(np.mean([pl[i] for i in ids if P[i]["band"] == k])), 1) for k in range(len(BANDS))],
                                         "rho": round(float(rho), 3), "p": float(f"{p:.2g}")}
json.dump(out, open(f"{HERE}/results/balanced.json", "w"), indent=1)
print("strata", len(STRATA))
print("\n(a) bands 2-3/4-5/6-7/8-9   raw -> balanced   rho_raw  rho|year,domain  mean within-year rho (neg years)")
for name in ORDER + ["agg3", "agg4"]:
    a = out["panel_a"][name]
    print(f"{name:11s} n={a['n']:4d} used={a['n_used']:4d} strata={a['n_strata']:2d} raw={a['raw_bands']} bal={a['balanced_bands']} rho={a['rho_raw']['rho']:+.3f} partial={a['rho_partial_year_domain']['rho']:+.3f} (p {a['rho_partial_year_domain']['p']}) wy={a['within_year']['mean']} ({a['within_year']['n_years_negative']}/{a['within_year']['n_years']})")
print("\n(b) FARS/reject/accept/oral raw -> balanced")
for name in ORDER:
    b = out["panel_b"][name]; print(f"{name:11s} n={b['n_by_group']} raw={b['raw_groups']} bal={b['balanced_groups']} rho_iclr={b['rho_iclr_raw']:+.3f} partial={b['rho_iclr_partial_year_domain']['rho']:+.3f}")
print("\n(c) items balanced means")
for k in out["panel_c"]:
    c = out["panel_c"][k]; print(f"{k:14s} " + " ".join(f"{g}={c['groups'][g]['mean']:.3f}(raw {c['groups'][g]['raw_mean']:.3f})" for g in GROUPS) + f" rho_iclr={c['rho_iclr_raw']:+.3f} partial={c['rho_iclr_partial_year_domain']['rho']:+.3f} rating_partial={c['rho_rating_partial_year_domain']['rho']:+.3f}")
print("\nsubsample", out["subsample"]["years_used"], out["subsample"]["n"])
for name, v in out["subsample"]["systems"].items(): print(f"  {name:11s} n={v['n']} bands={v['bands']} rho={v['rho']:+.3f} p={v['p']}")
