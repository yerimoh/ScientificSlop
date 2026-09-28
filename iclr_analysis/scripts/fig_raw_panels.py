r"""The same question without any normalisation. One small panel per system, each on its own raw scale.

Nothing is converted here. Binoculars keeps its perplexity ratio, DetectGPT its curvature z, the reviewers their
1 to 10 rating, and the slop score its share of failed units. The x axis is the ICLR review score in every panel,
and each panel is free to use whatever range its own numbers need. What the reader compares is the shape of the
line, which needs no common unit, and the dashed line marks where the AI papers of the benchmark sit on that same
scale, so each panel also says how far the whole ICLR range is from a typical AI paper.

Each paper's value is expressed against the mean of its own year before the review scores are pooled, because the
corpus drifts by year and two thirds of the papers at score 2 and 3 come from 2020 and 2025.
Outputs results/fig_raw_panels.{pdf,png}
"""
import csv, collections, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/bench165"
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]
TITLE = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS",
         "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "agg3": "SciSlop (ours)"}
UNIT = {"binoculars": "perplexity ratio", "detectgpt": "curvature z", "nts": "temperature sensitivity",
        "rev_b2h": "rating it gives", "rev_b3a": "rating it gives", "agg3": "share of failed units"}
COL = {"binoculars": "#2a78d6", "detectgpt": "#1baf7a", "nts": "#eda100",
       "rev_b2h": "#eb6834", "rev_b3a": "#e87ba4", "agg3": "#111111"}
MIN_CELL = 25

rows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv"))
        if r["group"] in ("reject", "accept", "oral") and r.get("rating") not in (None, "", "None")]
num = lambda r, c: None if r.get(c) in (None, "", "None") else float(r[c])

# where the benchmark's AI papers sit on each raw scale, for the reference line
items = json.load(open(f"{B}/items165.json"))["items"]
lab = {i["item_id"]: i["label"] for i in items}
AI_REF = {}
import glob
for k, key in (("binoculars", "binoculars"), ("detectgpt", "detectgpt"), ("nts", "nts")):
    v = [json.loads(l)[key] for f in glob.glob(f"{B}/results/{key}.jsonl") for l in open(f)
         if lab.get(json.loads(l)["id"]) == 1 and isinstance(json.loads(l).get(key), (int, float))]
    if v:
        AI_REF[k] = float(np.median(v))
ARCH = os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/Evaluation/02_baselines_B"
for k, sysname, dname in (("rev_b2h", "b2h", "B2h_cyclereviewer"), ("rev_b3a", "b3a", "B3a_ai_scientist")):
    v = []
    for f in glob.glob(f"{ARCH}/{dname}/runs/FA*/logs/*_{sysname}_R1_review.json"):
        x = ((json.load(open(f)) or {}).get("final") or {}).get("Overall")
        if isinstance(x, (int, float)):
            v.append(x)
    if v:
        AI_REF[k] = float(np.median(v))
brows = {}
for it in ("macro_redund", "xsec_ref", "citation"):
    for l in open(f"{B}/results/slop/{it}/papers.jsonl"):
        r = json.loads(l)
        brows.setdefault(r["id"], {"c": r["corpus"]})[it] = r.get("slop_score_agg") if it == "macro_redund" else r.get("slop_score")
v = [((x["macro_redund"] + x["xsec_ref"]) / 2 + x["citation"]) / 2 for x in brows.values()
     if x["c"] == "AI" and None not in (x.get("macro_redund"), x.get("xsec_ref"), x.get("citation"))]
AI_REF["agg3"] = float(np.median(v))

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.85), gridspec_kw={"wspace": 0.55})
for ax, k in zip(axes, ORDER):
    use = [r for r in rows if num(r, k) is not None]
    vals = np.array([num(r, k) for r in use], float)
    g = float(np.mean(vals))
    ym = collections.defaultdict(list)
    for r, v_ in zip(use, vals):
        ym[r["year"]].append(v_)
    ym = {y: float(np.mean(x)) for y, x in ym.items()}
    adj = np.array([v_ - ym[r["year"]] + g for r, v_ in zip(use, vals)], float)
    cells = collections.defaultdict(list)
    for r, v_ in zip(use, adj):
        cells[int(round(float(r["rating"])))].append(v_)
    pts = [(b, float(np.mean(v_))) for b, v_ in sorted(cells.items()) if len(v_) >= MIN_CELL]
    ax.plot([a for a, _ in pts], [b for _, b in pts], color=COL[k], lw=1.8, marker="o", ms=3.4,
            mec="white", mew=0.7, zorder=4)
    if k in AI_REF:
        ax.axhline(AI_REF[k], color="#bbbbbb", lw=0.8, ls="--", zorder=1)
        ax.annotate("AI papers", xy=(0.02, AI_REF[k]), xycoords=("axes fraction", "data"),
                    fontsize=5.2, color="#999999", va="bottom")
    rho = stats.spearmanr([float(r["rating"]) for r in use], adj)[0]
    ax.set_title(f"{TITLE[k]}", fontsize=7.5, pad=3)
    ax.set_xlabel(UNIT[k], fontsize=6.2, labelpad=2)
    ax.set_xticks([2, 5, 8])
    lo = min([b for _, b in pts] + ([AI_REF[k]] if k in AI_REF else []))
    hi = max([b for _, b in pts] + ([AI_REF[k]] if k in AI_REF else []))
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_ylim(lo - pad, hi + pad)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2, labelsize=6, color="#555555")
fig.text(0.5, 0.015, "ICLR review score", ha="center", fontsize=8)
fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.30)
fig.savefig(f"{HERE}/results/fig_raw_panels.pdf"); fig.savefig(f"{HERE}/results/fig_raw_panels.png", dpi=300)
print("saved fig_raw_panels; each panel keeps its own unit, the dashed line is the benchmark's AI papers")
for k in ORDER:
    print(f"  {TITLE[k]:16s} AI-paper reference line {AI_REF.get(k)}")
