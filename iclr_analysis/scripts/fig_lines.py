"""Extra line graphs for §5.2 paragraph 3.
fig_by_year      : x = ICLR year, one line per decision group, y = mean SciSlop (4 items); FARS as a dotted reference
fig_by_item_year : 2x2 panels, one per item, same layout
fig_rating_bins  : x = within-year rating quintile (1 low .. 5 high), one line per item, y = mean score
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = [json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.5, "pdf.fonttype": 42})
GCOL = {"reject": "#B04040", "accept": "#4A6FA5", "oral": "#3A8A5C"}
ITEMS = [("scislop_det4", "SciSlop (4 items)", "#222222", "-", "o"), ("macro_redund", "Macro redundancy", "#95627A", "-", "s"),
         ("xsec_ref", "Cross-section refs.", "#95627A", "--", "^"), ("citation", "Citation", "#E4959E", "-", "D"),
         ("evidence_gap", "Evidence gap", "#6D8A96", "-", "v")]
YEARS = list(range(2017, 2026))


def mean_ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return (np.nan, 0, 0)
    idx = np.random.randint(0, len(v), (1000, len(v))); b = v[idx].mean(1)
    return (v.mean(), v.mean() - np.percentile(b, 2.5), np.percentile(b, 97.5) - v.mean())


def by_year_panel(ax, key, title):
    fars = [p[key] for p in P if p["group"] == "FARS"]
    m = np.mean([x for x in fars if x is not None])
    ax.axhline(m, color="#999999", lw=0.8, ls=":"); ax.text(2017, m, "FARS (AI)", fontsize=6, color="#666666", va="bottom")
    for g, c in GCOL.items():
        ys, lo, hi = [], [], []
        for y in YEARS:
            a, l, h = mean_ci([p[key] for p in P if p.get("year") == y and p["group"] == g]); ys.append(a); lo.append(l); hi.append(h)
        ax.errorbar(YEARS, ys, yerr=[lo, hi], color=c, lw=1.0, marker="o", ms=2.5, capsize=1.5, elinewidth=0.5, label=g)
    ax.set_title(title, fontsize=8); ax.set_xticks(YEARS); ax.set_xticklabels([str(y)[2:] for y in YEARS], fontsize=6.5)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, labelsize=6.5)


np.random.seed(0)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
by_year_panel(ax, "scislop_det4", "SciSlop (4 items), mean per ICLR year")
ax.set_ylabel("slop score (mean, 95% CI)", fontsize=7); ax.set_xlabel("ICLR year", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3)
fig.tight_layout(); fig.savefig(f"{HERE}/results/fig_by_year.pdf"); fig.savefig(f"{HERE}/results/fig_by_year.png", dpi=300)

fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True)
for ax, (key, lab, *_) in zip(axes.flat, ITEMS[1:]):
    by_year_panel(ax, key, lab)
axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=3)
for ax in axes[:, 0]: ax.set_ylabel("slop score", fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/results/fig_by_item_year.pdf"); fig.savefig(f"{HERE}/results/fig_by_item_year.png", dpi=300)

iclr = [p for p in P if p["src"] == "iclr" and p["group"] in GCOL and p.get("rating_pct_in_year") is not None]
for p in iclr:
    p["q"] = int(round(p["rating"]))
QS = sorted({q for q in collections.Counter(p["q"] for p in iclr) if collections.Counter(p["q"] for p in iclr)[q] >= 20})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for key, lab, col, ls, mk in ITEMS:
    ys, lo, hi = [], [], []
    for q in QS:
        a, l, h = mean_ci([p[key] for p in iclr if p["q"] == q]); ys.append(a); lo.append(l); hi.append(h)
    ax.errorbar(QS, ys, yerr=[lo, hi], color=col, ls=ls, lw=1.3 if key == "scislop_det4" else 0.9, marker=mk, ms=3,
                mfc="white" if key == "xsec_ref" else col, capsize=1.5, elinewidth=0.5, label=lab)
    ax.annotate(lab, xy=(QS[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=5.6, va="center")
ns = collections.Counter(p["q"] for p in iclr)
ax.set_xticks(QS); ax.set_xticklabels([f"{q}\nn={ns[q]}" for q in QS], fontsize=6.5)
ax.set_xlabel("mean public review score (ICLR, 1-10)", fontsize=7); ax.set_ylabel("slop score (mean, 95% CI)", fontsize=7)
ax.set_ylim(0, 1); ax.yaxis.grid(True, color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(length=2, labelsize=6.5); ax.set_xlim(QS[0] - 0.4, QS[-1] + 0.4)
fig.subplots_adjust(left=0.15, right=0.72, top=0.97, bottom=0.25)
fig.savefig(f"{HERE}/results/fig_rating_bins.pdf"); fig.savefig(f"{HERE}/results/fig_rating_bins.png", dpi=300)
print("saved fig_by_year, fig_by_item_year, fig_rating_bins")
