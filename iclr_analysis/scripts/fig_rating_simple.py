"""Simple version of fig_rating_systems: x = review-score band (2-3, 4-5, 6-7, 8-9), y = mean score percentile.
Baselines = thin grey lines with a small end label; ours = one black line with 95% CI. No error bars on baselines.
Same data loading as fig_rating_systems.py (import it as a module by exec of its loading half).
"""
import json, os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(f"{HERE}/scripts/fig_rating_systems.py").read()
ns = {"__file__": f"{HERE}/scripts/fig_rating_systems.py"}
exec(src.split("# percentile (0-100)")[0], ns)          # loads S, DIR, LABEL, SRC, iclr
S, DIR, LABEL, SRC, iclr = ns["S"], ns["DIR"], ns["LABEL"], ns["SRC"], ns["iclr"]
from scipy import stats
BANDS = [(2, 3), (4, 5), (6, 7), (8, 9)]
PCT = {}
for name, d in S.items():
    ids = list(d); vals = np.array([d[i] * DIR[name] for i in ids], float)
    PCT[name] = dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))


def band_of(r):
    r = int(round(r))
    for k, (a, b) in enumerate(BANDS):
        if a <= r <= b:
            return k
    return None


rng = np.random.default_rng(0)
out = {}
for name in S:
    ids = list(S[name]); rho = stats.spearmanr([iclr[i]["rating"] for i in ids], [PCT[name][i] for i in ids])[0]
    ys, cis = [], []
    for k in range(len(BANDS)):
        yy = np.array([PCT[name][i] for i in ids if band_of(iclr[i]["rating"]) == k])
        ys.append(yy.mean() if len(yy) else np.nan)
        b = rng.choice(yy, (1000, len(yy))).mean(1) if len(yy) > 1 else np.array([np.nan])
        cis.append((np.percentile(b, 2.5), np.percentile(b, 97.5)))
    out[name] = {"label": LABEL[name], "n": len(ids), "rho": round(float(rho), 3), "band_means": [round(float(y), 1) for y in ys]}

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.0, 2.2))
x = list(range(len(BANDS)))
grey = [n for n in S if n != "scislop_det4" and not (n.endswith("_arch") and n[:-5] in S)]   # archive only where no fresh run
for j, name in enumerate(grey):
    ys = out[name]["band_means"]
    ax.plot(x, ys, color="#9a9a9a", lw=0.8, marker="o", ms=2, mfc="white", mew=0.6, zorder=2,
            label=f"baselines, one line each (n = {len(grey)})" if j == 0 else None)
ys = out["scislop_det4"]["band_means"]
ids = list(S["scislop_det4"])
lo = [ys[k] - np.percentile(rng.choice([PCT["scislop_det4"][i] for i in ids if band_of(iclr[i]["rating"]) == k], (1000, sum(band_of(iclr[i]["rating"]) == k for i in ids))).mean(1), 2.5) for k in x]
hi = [np.percentile(rng.choice([PCT["scislop_det4"][i] for i in ids if band_of(iclr[i]["rating"]) == k], (1000, sum(band_of(iclr[i]["rating"]) == k for i in ids))).mean(1), 97.5) - ys[k] for k in x]
ax.errorbar(x, ys, yerr=[lo, hi], color="#222222", lw=1.6, marker="o", ms=3.8, capsize=2, elinewidth=0.7, zorder=4)
ax.annotate("SciSlop (ours)", xy=(x[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color="#222222", va="center", fontweight="bold")
ax.legend(frameon=False, fontsize=6, loc="lower left", handlelength=1.6)
ax.axhline(50, color="#cccccc", lw=0.5, ls=":", zorder=1)
ax.set_xticks(x); ax.set_xticklabels([f"{a}–{b}" for a, b in BANDS], fontsize=7)
ax.set_xlabel("mean review score (ICLR)", fontsize=7)
ax.set_ylabel("score percentile", fontsize=7)
ax.set_ylim(30, 70); ax.set_yticks([30, 40, 50, 60, 70])
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(length=2, labelsize=7); ax.set_xlim(-0.2, len(BANDS) - 0.8)
fig.subplots_adjust(left=0.17, right=0.74, top=0.96, bottom=0.2)
suf = "_archive" if "--archive" in sys.argv else ""
fig.savefig(f"{HERE}/results/fig_rating_simple{suf}.pdf"); fig.savefig(f"{HERE}/results/fig_rating_simple{suf}.png", dpi=300)
json.dump({"bands": BANDS, "systems": out}, open(f"{HERE}/results/rating_simple{suf}.json", "w"), indent=1)
for name, v in out.items():
    print(f"{name:18s} n={v['n']:5d} rho={v['rho']:+.3f} bands={v['band_means']}")
