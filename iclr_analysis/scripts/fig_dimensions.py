r"""What each system is actually tracking, on the two axes the review data offers.

(a) The four scores an ICLR reviewer gives. A paper gets an overall rating and, since 2022, a separate soundness,
    presentation and contribution score. Each system's score is correlated with each of the four inside ICLR 2026,
    the one year that carries all four for every paper. A filled ring marks a correlation that clears p < 0.05.
    The sign is oriented so that a negative value means a better-scored paper looks less AI-like to that system.

(b) The same question asked without any scale at all. Inside each ICLR year the papers are split into the top and
    bottom third of that year's rating, and the bar is how much more often a highly rated paper lands in the upper
    half of that system's own distribution for that year. Zero means no relation. Nine years, one bar each.

Reads results/scores_2026.csv and results/year_rate.json. Outputs results/fig_dimensions.{pdf,png} and
results/dimensions.json
"""
import csv, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = {"binoculars": -1, "detectgpt": +1, "nts": +1, "pangram_fraction_ai": +1, "rev_b2h": -1, "rev_b3a": -1,
       "macro_redund": +1, "xsec_ref": +1, "citation": +1, "evidence_gap": +1, "agg3": +1}
ROWS = [("binoculars", "Binoculars"), ("detectgpt", "DetectGPT"), ("nts", "NTS"),
        ("pangram_fraction_ai", "Pangram"), ("rev_b2h", "CycleReviewer"), ("rev_b3a", "AI Scientist"),
        ("agg3", "SciSlop (ours)")]
DIMS = [("rating", "overall"), ("soundness", "soundness"), ("presentation", "presentation"), ("contribution", "contribution")]
rows = list(csv.DictReader(open(f"{HERE}/results/scores_2026.csv")))
num = lambda r, c: None if r.get(c) in (None, "", "None") else float(r[c])

OUT = {"note": __doc__, "n_papers": len(rows), "cells": {}}
cell = {}
for k, lab in ROWS:
    sel = [r for r in rows if num(r, k) is not None]
    OUT["cells"][k] = {"n": len(sel)}
    for dim, _ in DIMS:
        xy = [(num(r, k) * DIR[k], num(r, dim)) for r in sel if num(r, dim) is not None]
        if len(xy) < 60:
            continue
        rho, p = stats.spearmanr([a for a, _ in xy], [b for _, b in xy])
        cell[(k, dim)] = (float(rho), float(p))
        OUT["cells"][k][dim] = {"rho": round(float(rho), 3), "p": float(f"{p:.3g}"), "n": len(xy)}

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.05, "width_ratios": [1.0, 1]})

# ---- (a) dot matrix
ys = list(range(len(ROWS)))[::-1]
for yi, (k, lab) in zip(ys, ROWS):
    for xi, (dim, _) in enumerate(DIMS):
        if (k, dim) not in cell:
            continue
        rho, p = cell[(k, dim)]
        c = "#B04040" if rho > 0 else "#2C6E8F"
        s = 18 + 900 * abs(rho)
        ax1.scatter([xi], [yi], s=s, facecolor=c if p < 0.05 else "white", edgecolor=c,
                    linewidth=0.9, alpha=0.95 if p < 0.05 else 0.8, zorder=3)
ax1.set_xticks(range(len(DIMS))); ax1.set_xticklabels([d[1] for d in DIMS], fontsize=7)
ax1.set_yticks(ys); ax1.set_yticklabels([lab for _, lab in ROWS], fontsize=7)
ax1.set_xlim(-0.6, len(DIMS) - 0.4); ax1.set_ylim(-0.8, len(ROWS) - 0.2)
ax1.axhline(0.5, color="#dddddd", lw=0.7)
ax1.set_xlabel("the four scores an ICLR reviewer gives", fontsize=7.6, labelpad=4)
for sp in ("top", "right", "left"):
    ax1.spines[sp].set_visible(False)
ax1.spines["bottom"].set_color("#555555")
ax1.tick_params(length=0, labelsize=7, color="#555555")
ax1.scatter([], [], s=18 + 900 * 0.1, facecolor="#2C6E8F", edgecolor="#2C6E8F", label="better score, less AI-like")
ax1.scatter([], [], s=18 + 900 * 0.1, facecolor="#B04040", edgecolor="#B04040", label="better score, more AI-like")
ax1.scatter([], [], s=18 + 900 * 0.1, facecolor="white", edgecolor="#888888", label="not significant")
ax1.legend(frameon=False, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.45, -0.30), ncol=3,
           handletextpad=0.3, columnspacing=0.9)

# ---- (b) per year, can the system tell a rejected paper from an accepted one
import itertools
yrows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv")) if r["group"] in ("reject", "accept", "oral")]
YDIR = {"binoculars": -1, "detectgpt": +1, "nts": +1, "agg3": +1}
YSTY = {"binoculars": ("#2a78d6", "o", "Binoculars"), "detectgpt": ("#1baf7a", "s", "DetectGPT"),
        "nts": ("#eda100", "D", "NTS"), "agg3": ("#111111", "o", "SciSlop (ours)")}
def auroc(p, n):
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(p, n)) / (len(p) * len(n))
years = [str(y) for y in range(2017, 2026)]
OUT["accept_auroc_by_year"] = {}
for k, d in YDIR.items():
    xs, ys = [], []
    for i, y in enumerate(years):
        rej = [num(r, k) * d for r in yrows if r["year"] == y and r["group"] == "reject" and num(r, k) is not None]
        acc = [num(r, k) * d for r in yrows if r["year"] == y and r["group"] in ("accept", "oral") and num(r, k) is not None]
        if len(rej) >= 15 and len(acc) >= 15:
            xs.append(i); ys.append(auroc(rej, acc))
    OUT["accept_auroc_by_year"][k] = {years[i]: round(v, 3) for i, v in zip(xs, ys)}
    c, mk, lab = YSTY[k]; ours = k == "agg3"
    ax2.plot(xs, ys, color=c, lw=2.2 if ours else 1.1, marker=mk, ms=4.6 if ours else 3.0,
             mec="white" if ours else c, mew=0.8 if ours else 0.5, zorder=5 if ours else 3, label=lab)
ax2.axhline(0.5, color="#999999", lw=0.8)
ax2.text(0.05, 0.505, "cannot tell them apart", fontsize=5.6, color="#999999", va="bottom")
ax2.set_xticks(range(len(years))); ax2.set_xticklabels([y[2:] for y in years], fontsize=7)
ax2.set_xlabel("ICLR year", fontsize=7.6)
ax2.set_ylabel("tells a rejected paper from an accepted one (AUROC)", fontsize=7.2)
ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()
ax2.set_ylim(0.38, 0.76)
for sp in ("top", "left"):
    ax2.spines[sp].set_visible(False)
ax2.spines["right"].set_color("#555555"); ax2.spines["bottom"].set_color("#555555")
ax2.tick_params(length=2.5, labelsize=7, color="#555555")
ax2.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4,
           handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
fig.subplots_adjust(left=0.135, right=0.905, top=0.97, bottom=0.26)
fig.savefig(f"{HERE}/results/fig_dimensions.pdf"); fig.savefig(f"{HERE}/results/fig_dimensions.png", dpi=300)
json.dump(OUT, open(f"{HERE}/results/dimensions.json", "w"), indent=1)
print("Accept/reject discrimination AUROC by year:", {k: v for k, v in OUT["accept_auroc_by_year"].items()})
print(f"{'system':20s} " + " ".join(f"{d[1][:12]:>13s}" for d in DIMS))
for k, lab in ROWS:
    line = []
    for dim, _ in DIMS:
        if (k, dim) in cell:
            rho, p = cell[(k, dim)]
            line.append(f"{rho:+8.3f}{'*' if p < 0.05 else ' '}")
        else:
            line.append(f"{'-':>9s}")
    print(f"{lab:20s} " + " ".join(f"{x:>13s}" for x in line))
