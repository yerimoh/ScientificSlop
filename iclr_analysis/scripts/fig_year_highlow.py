"""Year analysis (0916 night, user request): for every ICLR year, split the papers into the top and bottom third of
that year's public review scores and plot each system's AI-likeness percentile for the two groups, year by year.
y = percentile of the system's score within its whole ICLR sample (so year drift stays visible), AI-oriented.
Claim under test: our slop score is lower for high-scored papers in every year, text detectors show no such gap
(or the opposite), automated reviewers show it too. Also an absolute-threshold variant (low <= 4, high >= 7).
Outputs results/year_highlow.json, results/fig_year_highlow.{pdf,png} (small multiples, one per system),
results/fig_year_gap.{pdf,png} (high minus low gap, one line per system)."""
import json, os, sys, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import load, HERE, ORDER, STYLE, ITEMS
rng = np.random.default_rng(0)
P, S, DIR, LABEL = load("agg5")
YEARS = list(range(2017, 2026))
LABELS = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "ours": "SciSlop (ours, 5 measures)", "agg3": "SciSlop (ours, 3 deterministic)"}
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v.get("rating") is not None and v["year"] <= 2025}
# within-year terciles over the full filtered ICLR sample (same split for every system)
TER = {}
for y in YEARS:
    ids = [k for k, v in iclr.items() if v["year"] == y]
    r = np.array([iclr[k]["rating"] for k in ids]); lo, hi = np.percentile(r, [100 / 3, 200 / 3])
    for k in ids:
        TER[k] = "low" if iclr[k]["rating"] <= lo else ("high" if iclr[k]["rating"] >= hi else "mid")
ABS = {k: ("low" if v["rating"] <= 4.5 else ("high" if v["rating"] >= 6.5 else "mid")) for k, v in iclr.items()}   # rounded 1-4 vs 7-10


def mean_ci(v, n=2000):
    v = np.asarray(v, float)
    if len(v) < 3: return None
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"n": int(len(v)), "mean": round(float(v.mean()), 2), "ci": [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]}


def gap_ci(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3: return None
    g = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    p = stats.mannwhitneyu(a, b).pvalue
    return {"gap": round(float(a.mean() - b.mean()), 2), "ci": [round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)], "mwu_p": float(f"{p:.2g}")}


out = {"note": __doc__, "split": {}, "systems": {}, "items_raw": {}}
for y in YEARS:
    ids = [k for k in iclr if iclr[k]["year"] == y]
    out["split"][y] = {"n": len(ids), "tercile": {t: {"n": sum(1 for k in ids if TER[k] == t), "mean_rating": round(float(np.mean([iclr[k]["rating"] for k in ids if TER[k] == t])), 2)} for t in ("low", "high")},
                       "absolute": {t: {"n": sum(1 for k in ids if ABS[k] == t)} for t in ("low", "high")}}
for name in ORDER + ["agg3"]:
    ids = [k for k in S[name] if k in iclr]
    vals = np.array([S[name][k] * DIR[name] for k in ids]); pct = dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))
    d = {"label": LABELS[name], "n": len(ids), "tercile": {}, "absolute": {}}
    for split_name, SPLIT in (("tercile", TER), ("absolute", ABS)):
        gaps = []
        for y in YEARS:
            lo = [pct[k] for k in ids if iclr[k]["year"] == y and SPLIT[k] == "low"]; hi = [pct[k] for k in ids if iclr[k]["year"] == y and SPLIT[k] == "high"]
            d[split_name][y] = {"low": mean_ci(lo), "high": mean_ci(hi), "gap": gap_ci(hi, lo)}
            if d[split_name][y]["gap"]: gaps.append(d[split_name][y]["gap"]["gap"])
        # pooled gap with year removed: mean of yearly gaps, and sign count
        d[split_name]["summary"] = {"mean_gap": round(float(np.mean(gaps)), 2), "n_years": len(gaps), "n_years_negative": sum(1 for g in gaps if g < 0),
                                    "n_years_negative_p05": sum(1 for y in YEARS if d[split_name][y]["gap"] and d[split_name][y]["gap"]["gap"] < 0 and d[split_name][y]["gap"]["mwu_p"] < 0.05),
                                    "n_years_positive_p05": sum(1 for y in YEARS if d[split_name][y]["gap"] and d[split_name][y]["gap"]["gap"] > 0 and d[split_name][y]["gap"]["mwu_p"] < 0.05)}
    out["systems"][name] = d
# raw item scores, high vs low tercile per year (for the report: which measures carry the gap)
for k, lab, *_ in ITEMS + [("scislop5", "SciSlop 5", None, None, None), ("scislop_det4", "SciSlop det4", None, None, None)]:
    d = {}
    for y in YEARS:
        lo = [P[i][k] for i in iclr if P[i]["year"] == y and TER[i] == "low" and P[i].get(k) is not None]; hi = [P[i][k] for i in iclr if P[i]["year"] == y and TER[i] == "high" and P[i].get(k) is not None]
        d[y] = {"low": round(float(np.mean(lo)), 4) if lo else None, "high": round(float(np.mean(hi)), 4) if hi else None, "n": [len(lo), len(hi)]}
    gaps = [d[y]["high"] - d[y]["low"] for y in YEARS if d[y]["low"] is not None and d[y]["high"] is not None]
    d["summary"] = {"mean_gap": round(float(np.mean(gaps)), 4), "n_years_negative": sum(1 for g in gaps if g < 0), "n_years": len(gaps)}
    out["items_raw"][k] = d
json.dump(out, open(f"{HERE}/results/year_highlow.json", "w"), indent=1)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
C_HI, C_LO = "#4A6FA5", "#B04040"
xs = list(range(len(YEARS)))


def draw_small_multiples(split_name, suffix):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ORDER):
        d = out["systems"][name][split_name]
        for t, col, mk, lab in (("high", C_HI, "o", "top third of review scores"), ("low", C_LO, "s", "bottom third of review scores")):
            y = [d[yr][t]["mean"] if d[yr][t] else np.nan for yr in YEARS]
            lo = [d[yr][t]["mean"] - d[yr][t]["ci"][0] if d[yr][t] else 0 for yr in YEARS]; hi = [d[yr][t]["ci"][1] - d[yr][t]["mean"] if d[yr][t] else 0 for yr in YEARS]
            ax.errorbar(xs, y, yerr=[lo, hi], color=col, lw=1.3 if name == "ours" else 1.0, marker=mk, ms=3, capsize=1.3, elinewidth=0.5, label=lab if name == "binoculars" else None, zorder=3)
        ax.axhline(50, color="#cccccc", lw=0.6, ls=":", zorder=1)
        sm = d["summary"]
        ax.set_title(LABELS[name].replace(", 5 measures", ""), fontsize=8, pad=3, fontweight="bold" if name == "ours" else "normal")
        ax.text(0.02, 0.04, f"high − low: {sm['mean_gap']:+.0f} pts, {sm['n_years_negative']}/{sm['n_years']} years negative", transform=ax.transAxes, fontsize=6, color="#333333")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=6.8, color="#555555")
    axes[0, 0].set_ylim(15, 85); axes[0, 0].set_yticks([20, 35, 50, 65, 80])
    for ax in axes[1]: ax.set_xticks(xs); ax.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS]); ax.set_xlabel("ICLR year", fontsize=7.2)
    for ax in axes[:, 0]: ax.set_ylabel("AI-likeness (percentile)", fontsize=7.2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.17, hspace=0.35, wspace=0.12)
    fig.savefig(f"{HERE}/results/fig_year_highlow{suffix}.pdf"); fig.savefig(f"{HERE}/results/fig_year_highlow{suffix}.png", dpi=300); plt.close(fig)


draw_small_multiples("tercile", "")
draw_small_multiples("absolute", "_abs")

# gap figure: one line per system
fig, ax = plt.subplots(figsize=(5.4, 2.6))
for name in ORDER + ["agg3"]:
    d = out["systems"][name]["tercile"]; c, mk = STYLE["ours" if name == "agg3" else name]; ours = name in ("ours", "agg3")
    y = [d[yr]["gap"]["gap"] if d[yr]["gap"] else np.nan for yr in YEARS]
    ax.plot(xs, y, color=c, lw=2.0 if ours else 1.0, ls="--" if name == "agg3" else "-", marker=mk, ms=4.5 if ours else 3, mfc="white" if name == "agg3" else c, mec=c if name == "agg3" else ("white" if ours else c), mew=0.7,
            zorder=6 if name == "ours" else (5 if ours else 3), label=f"{LABELS[name]}  ({d['summary']['mean_gap']:+.0f}, {d['summary']['n_years_negative']}/{d['summary']['n_years']})")
ax.axhline(0, color="#888888", lw=0.7, zorder=1)
ax.set_xticks(xs); ax.set_xticklabels([f"'{str(y)[2:]}" for y in YEARS], fontsize=7); ax.set_xlabel("ICLR year", fontsize=7.5)
ax.set_ylabel("AI-likeness gap, top − bottom third\n(percentile points)", fontsize=7.2)
ax.annotate("high-scored papers look more AI-like", xy=(0.01, 0.985), xycoords="axes fraction", fontsize=5.8, color="#555555", va="top", style="italic")
ax.annotate("high-scored papers look less AI-like", xy=(0.01, 0.015), xycoords="axes fraction", fontsize=5.8, color="#555555", va="bottom", style="italic")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=7, color="#555555")
ax.set_ylim(-45, 30)
ax.legend(frameon=False, fontsize=6.2, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.0, labelspacing=0.45, title="system  (mean gap, years negative)", title_fontsize=6.2)
fig.tight_layout(pad=0.4)
fig.savefig(f"{HERE}/results/fig_year_gap.pdf"); fig.savefig(f"{HERE}/results/fig_year_gap.png", dpi=300)
print("split per year:", {y: (v["tercile"]["low"]["n"], v["tercile"]["high"]["n"], v["tercile"]["low"]["mean_rating"], v["tercile"]["high"]["mean_rating"]) for y, v in out["split"].items()})
for name in ORDER + ["agg3"]:
    for sn in ("tercile", "absolute"):
        d = out["systems"][name][sn]; sm = d["summary"]
        print(f"{name:11s} {sn:8s} n={out['systems'][name]['n']:4d} gaps=" + " ".join(f"{yr}:{d[yr]['gap']['gap']:+5.1f}" if d[yr]["gap"] else f"{yr}:  n/a" for yr in YEARS) + f" | mean {sm['mean_gap']:+.1f} neg {sm['n_years_negative']}/{sm['n_years']} sig- {sm['n_years_negative_p05']} sig+ {sm['n_years_positive_p05']}")
print("\nraw items, high - low tercile per year (mean gap, years negative):")
for k, d in out["items_raw"].items():
    print(f"  {k:14s} mean gap {d['summary']['mean_gap']:+.4f}  neg {d['summary']['n_years_negative']}/{d['summary']['n_years']}  " + " ".join(f"{y}:{d[y]['high']-d[y]['low']:+.3f}" for y in YEARS if d[y]['low'] is not None and d[y]['high'] is not None))
