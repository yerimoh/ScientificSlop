"""Fig 3 of §5.2: mean slop per group FARS / ICLR reject / ICLR accept / ICLR oral, one line per
deterministic item plus their aggregate. Reads results/stairs.json. One panel, printed at 0.5\textwidth.
Colours = the paper's plane colours (iclr2027_conference.tex): Structure #95627A, Argument #E4959E, Artifacts #6D8A96.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{HERE}/results/stairs.json"))
key = sys.argv[1] if len(sys.argv) > 1 else "main"
M = S["main" if key == "all6" else key]
GROUPS = ["FARS", "reject", "accept", "oral"]
XLAB = ["FARS\n(AI)", "ICLR\nreject", "ICLR\naccept", "ICLR\noral"]

WIDTH_IN = 2.75   # printed at 0.5\textwidth (5.5in) => font sizes below are printed points
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "pdf.fonttype": 42})
LINES = [  # key, label, colour, linestyle, marker
    ("scislop_det4", "SciSlop (4 items)", "#222222", "-", "o"),
    ("macro_redund", "Macro redundancy", "#95627A", "-", "s"),
    ("xsec_ref", "Cross-section refs.", "#95627A", "--", "^"),
    ("citation", "Citation", "#E4959E", "-", "D"),
    ("evidence_gap", "Evidence gap", "#6D8A96", "-", "v"),
]
if key == "all6":            # 0916: main stairs plus argument_graph, fig_exposition and the 5-item aggregate
    M = S["main"]
    LINES = [("scislop5", "SciSlop (5 items)", "#222222", "-", "o"), ("scislop_det4", "SciSlop (4 items)", "#777777", "--", "o")] + LINES[1:] + [
        ("argument_graph", "Argument graph", "#E4959E", "--", "P"), ("fig_exposition", "Figure exposition", "#6D8A96", ":", "X")]
    LINES = [l for l in LINES if l[0] in M["groups"]]
fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.15))
x = list(range(4))
for k, lab, col, ls, mk in LINES:
    g = M["groups"][k]
    y = [g[gr]["mean"] for gr in GROUPS]
    lo = [y[i] - g[gr]["ci95"][0] for i, gr in enumerate(GROUPS)]
    hi = [g[gr]["ci95"][1] - y[i] for i, gr in enumerate(GROUPS)]
    lw = 1.3 if k in ("scislop_det4", "scislop5") else 0.9
    ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, lw=lw, marker=mk, ms=3.2 if k != "scislop_det4" else 3.8,
                mfc="white" if k == "xsec_ref" else col, mew=0.8, capsize=1.5, elinewidth=0.6, label=lab, zorder=3 if k == "scislop_det4" else 2)
    ax.annotate(lab, xy=(3, y[3]), xytext=(4, 0), textcoords="offset points", fontsize=5.6, color="#222222", va="center", ha="left")
ax.axvline(0.5, color="#bbbbbb", lw=0.5, ls=":", zorder=1)
ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=6.5)
ns = M["n_by_group"]
for i, gr in enumerate(GROUPS):
    ax.text(i, -0.22, f"n={ns[gr]}", ha="center", va="top", fontsize=5.4, color="#555555", transform=ax.get_xaxis_transform())
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("slop score (mean, 95% CI)", fontsize=6.5)
ax.yaxis.grid(True, color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=2, labelsize=6)
ax.set_xlim(-0.35, 3.35)
fig.subplots_adjust(left=0.16, right=0.70, top=0.97, bottom=0.24)
suffix = "" if key == "main" else ("_6items" if key == "all6" else f"_{key}")
for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/results/fig_review_stairs{suffix}.{ext}", dpi=300)
print("saved", f"{HERE}/results/fig_review_stairs{suffix}.pdf")
