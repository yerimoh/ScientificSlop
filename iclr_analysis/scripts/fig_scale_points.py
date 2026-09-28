r"""The review-score panel drawn on ICLR's own rating levels, 1 3 5 6 8 10.

Why this axis. A reviewer picks one level of the ICLR scale and a paper's score is the mean of three or four such
levels, so the score falls between them. Each paper is placed at the nearest whole point, which keeps every step
from 2 to 9 with 25 to 50 papers in each. The scale itself changed over the years, a matter the year by year panel
of the main figure handles by staying inside each year; the ordering it shows is the same one this axis assumes.

Why no 1 and no 10. A mean reaches an end only when every reviewer picks that end, which leaves 11 papers at score 1
and one at score 10, below the minimum cell size.

The corpus also drifts by year, so each paper's value is expressed against the mean of its own year before the score
levels are pooled. Without that step the low levels read as a year effect, since two thirds of the papers at 2 and 3
come from 2020 and 2025.

Every system is put on the same axis by calibration, not by ranking. Each system is fitted once on SciSlopBench,
where the answer is known, so its raw score becomes the probability it assigns to AI authorship. Ranking would make
every system uniform and hide how far apart the baselines are; a calibrated probability keeps a system that cannot
separate the two classes pinned near one half.
Outputs results/fig_scale_points.{pdf,png} and results/scale_points.json
"""
import csv, json, os, sys, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = [1, 3, 5, 6, 8, 10]
MIN_CELL = 25
DIR = {"binoculars": -1, "detectgpt": +1, "nts": +1, "rev_b2h": -1, "rev_b3a": -1, "agg3": +1}
STYLE = {"binoculars": ("#2a78d6", "o", "Binoculars"), "detectgpt": ("#1baf7a", "s", "DetectGPT"),
         "nts": ("#eda100", "D", "NTS"), "rev_b2h": ("#eb6834", "v", "CycleReviewer"),
         "rev_b3a": ("#e87ba4", "P", "AI Scientist"), "agg3": ("#111111", "o", "SciSlop (ours)")}
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]

rows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv"))
        if r["group"] in ("reject", "accept", "oral") and r.get("rating") not in (None, "", "None")]
num = lambda r, c: None if r.get(c) in (None, "", "None") else float(r[c])
common = [r for r in rows if all(num(r, k) is not None for k in ORDER)]
# Each system is drawn on every paper it scored, not on the intersection. Restricting all six to the 282 papers the
# reviewers reached leaves 30 papers a score level, where the mean of a probability with a spread of 25 points
# carries a standard error of about 5, and the low levels then wander. The count each line rests on is in the legend.
FULL = {k: [r for r in rows if num(r, k) is not None] for k in ORDER}
# A paper's score is the mean of three or four reviewer levels, so it lands between them. Rounding to the nearest
# whole point keeps every step of the scale, including 4 and 7, and leaves 25 to 50 papers in each step. Snapping
# onto the six ICLR levels instead would drop 4 and 7 and pile the papers into four uneven cells.
snap = lambda x: int(round(x))

OUT = {"note": __doc__, "n_corpus": len(rows), "n_common": len(common), "min_cell": MIN_CELL, "levels": {},
       "papers_per_level_corpus": dict(collections.Counter(snap(float(r["rating"])) for r in rows)),
       "papers_per_level_common": dict(collections.Counter(snap(float(r["rating"])) for r in common))}
CAL = json.load(open(f"{HERE}/results/calibration.json"))["calibration"]
AXIS = "def" if "--def" in sys.argv else "pai"
# definitional axis: a score whose own definition fixes both ends. Ours is failed units over checked units, already
# in [0, 1]. A reviewer's Overall lives on the 1-10 rating scale, so (10 - x) / 9 puts its AI-like end at 1. A
# perplexity ratio and a z score have no definitional end, so the three detectors cannot appear on this axis.
DEFN = {"agg3": lambda v: v, "rev_b2h": lambda v: (10.0 - v) / 9.0, "rev_b3a": lambda v: (10.0 - v) / 9.0}
def p_ai(k, v):
    c = CAL[k]
    return float(1.0 / (1.0 + np.exp(-(c["a"] + c["b"] * (v - c["mu"]) / c["sd"]))))
if AXIS == "def":
    ORDER = [k for k in ORDER if k in DEFN]
for k in ORDER:
    use = FULL[k]
    if AXIS == "def":
        pct = np.array([100 * float(np.clip(DEFN[k](num(r, k)), 0, 1)) for r in use], float)
    else:
        pct = np.array([100 * p_ai(k, num(r, k)) for r in use], float)
    # the corpus drifts by year, so each paper is expressed against its own year before the levels are pooled
    gmean = float(np.mean(pct))
    ymean = {}
    for r, p in zip(use, pct):
        ymean.setdefault(r["year"], []).append(p)
    ymean = {y: float(np.mean(v)) for y, v in ymean.items()}
    pct = np.array([p - ymean[r["year"]] + gmean for r, p in zip(use, pct)], float)
    cells = collections.defaultdict(list)
    for r, p in zip(use, pct):
        cells[snap(float(r["rating"]))].append(p)
    OUT["levels"][k] = {str(s): {"n": len(v), "mean_percentile": round(float(np.mean(v)), 1), "sd": round(float(np.std(v)), 1)}
                        for s, v in sorted(cells.items()) if len(v) >= MIN_CELL}
    xs = [float(r["rating"]) for r in use]
    OUT["levels"][k]["_rho"] = round(float(stats.spearmanr(xs, pct)[0]), 3)
    OUT["levels"][k]["_p"] = float(f"{stats.spearmanr(xs, pct)[1]:.3g}")
    OUT["levels"][k]["_n"] = len(use)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.0, 2.5))
drawn = sorted({int(s) for k in ORDER for s in OUT["levels"][k] if not s.startswith("_")})
for k in ORDER:
    pts = [(int(s), v["mean_percentile"]) for s, v in OUT["levels"][k].items() if not s.startswith("_")]
    pts.sort()
    if len(pts) < 3:
        continue
    c, mk, lab = STYLE[k]
    ours = k == "agg3"
    ax.plot([a for a, _ in pts], [b for _, b in pts], color=c, lw=2.4 if ours else 1.1, marker=mk,
            ms=5.5 if ours else 3.4, mec="white" if ours else c, mew=0.9 if ours else 0.5,
            zorder=6 if ours else 3, label=lab, clip_on=False)
if AXIS == "pai":
    ax.axhline(50, color="#d5d5d5", lw=0.6, ls=":", zorder=1)
    ax.text(min(drawn) - 0.2, 50.6, "coin flip", fontsize=5.6, color="#999999", va="bottom", ha="left")
ax.set_xticks(drawn)
ax.set_xlabel("ICLR review score", fontsize=8)
ax.set_ylabel("slop score / inverted review score (%)" if AXIS == "def" else "how AI-written the system says the paper is (%)", fontsize=8)
lo = min(v["mean_percentile"] for k in ORDER for s, v in OUT["levels"][k].items() if not s.startswith("_"))
hi = max(v["mean_percentile"] for k in ORDER for s, v in OUT["levels"][k].items() if not s.startswith("_"))
ax.set_ylim(max(0, np.floor((lo - 4) / 10) * 10), min(100, np.ceil((hi + 4) / 10) * 10))
ax.set_xlim(min(drawn) - 0.3, max(drawn) + 3.0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
ax.tick_params(length=2.5, labelsize=7.5, color="#555555")
# name each line at its right-hand end instead of a legend box, so nothing sits on top of the curves
END = {"rev_b2h": (5, 14), "binoculars": (5, 6), "rev_b3a": (5, -2), "nts": (5, -10), "detectgpt": (5, -18), "agg3": (5, 0)}
for k in ORDER:
    pts = sorted((int(b), v["mean_percentile"]) for b, v in OUT["levels"][k].items() if not b.startswith("_"))
    if len(pts) < 3:
        continue
    c, mk, lab = STYLE[k]
    off = END.get(k, (5, 0))
    ax.annotate(lab, xy=pts[-1], xytext=off, textcoords="offset points", fontsize=6.4,
                color="#111111" if k == "agg3" else c, va="center", ha="left",
                fontweight="bold" if k == "agg3" else "normal",
                arrowprops=None if abs(off[1]) < 4 else dict(arrowstyle="-", color=c, lw=0.4, shrinkA=1, shrinkB=1))
fig.tight_layout(pad=0.4)
suf = "_def" if AXIS == "def" else ""
fig.savefig(f"{HERE}/results/fig_scale_points{suf}.pdf"); fig.savefig(f"{HERE}/results/fig_scale_points{suf}.png", dpi=300)
json.dump(OUT, open(f"{HERE}/results/scale_points{suf}.json", "w"), indent=1)
print(f"corpus {len(rows)}, every system scored {len(common)}")
print("papers per level, corpus:", OUT["papers_per_level_corpus"])
print("papers per level, common:", OUT["papers_per_level_common"])
for k in ORDER:
    d = OUT["levels"][k]
    print(f"  {STYLE[k][2]:16s} rho={d['_rho']:+.3f} p={d['_p']:.3g}  " +
          " ".join(f"{s}:{v['mean_percentile']:.0f}(n{v['n']})" for s, v in sorted(d.items(), key=lambda kv: kv[0]) if not s.startswith("_")))
