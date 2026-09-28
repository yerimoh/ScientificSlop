"""Provenance x quality map (0916 night, user asked for a more interesting analysis).
One point per system. x = how the system tracks reviewer-judged quality inside human papers: partial Spearman of the
system's AI-oriented score against the public ICLR rating, year x domain removed (results/balanced.json). Negative =
highly rated papers look less AI-like. y = how the system separates provenance: AUROC on the 143 SciSlopBench pairs
(Table 2 values from results/pairwise165.json and the paper table; SciSlop aggregates recomputed here from the bench rows).
Reading: text detectors separate provenance but do not track quality (or track it the wrong way), automated reviewers
track quality but not provenance, the slop score is the only system in the upper-left corner.
Outputs results/prov_quality_map.json, results/fig_prov_quality_map.{pdf,png}."""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import HERE, STYLE, mean_skip
BAL = json.load(open(f"{HERE}/results/balanced.json"))["panel_a"]
PW = json.load(open(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/bench165/results/pairwise165.json"))["pair_metrics"]
# provenance AUROC: baselines from pairwise165.json / paper Table 2 (NTS 0.624 is the table value), ours from bench rows
AUROC = {"binoculars": PW["binoculars"]["all"]["auroc"], "detectgpt": PW["detectgpt"]["all"]["auroc"], "nts": 0.624,
         "rev_b2h": PW["rev_b2h"]["all"]["auroc"], "rev_b3a": PW["rev_b3a"]["all"]["auroc"]}
rows = [json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")]
bench = [r for r in rows if r["src"] in ("bench_AI", "bench_HU")]
def agg3(v):
    if None in (v.get("macro_redund"), v.get("xsec_ref"), v.get("citation")): return None
    return mean_skip([mean_skip([v["macro_redund"], v["xsec_ref"]]), v["citation"]])
for key, fn in [("ours", lambda v: v.get("scislop5")), ("agg3", agg3)]:
    y = [(1 if r["src"] == "bench_AI" else 0, fn(r)) for r in bench if fn(r) is not None]
    AUROC[key] = round(float(roc_auc_score([a for a, _ in y], [b for _, b in y])), 3)
LAB = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "ours": "SciSlop (5 measures)", "agg3": "SciSlop (3 deterministic)"}
out = {"note": __doc__, "systems": {k: {"label": LAB[k], "quality_partial_rho": BAL[k]["rho_partial_year_domain"]["rho"], "quality_p": BAL[k]["rho_partial_year_domain"]["p"],
                                          "quality_within_year_mean_rho": BAL[k]["within_year"]["mean"], "provenance_auroc": AUROC[k]} for k in LAB}}
json.dump(out, open(f"{HERE}/results/prov_quality_map.json", "w"), indent=1)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.axvline(0, color="#bbbbbb", lw=0.6, ls=":", zorder=1); ax.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":", zorder=1)
ax.axvspan(-0.45, 0, ymin=(0.75 - 0.4) / 0.6, ymax=1, color="#f2f2f2", zorder=0)
OFF = {"binoculars": (5, 4), "detectgpt": (-6, 0), "nts": (5, 2), "rev_b2h": (-4, -9), "rev_b3a": (5, 4), "ours": (6, -3), "agg3": (6, 4)}
for k, s in out["systems"].items():
    c, mk = STYLE["ours" if k == "agg3" else k]; ours = k in ("ours", "agg3")
    ax.scatter(s["quality_partial_rho"], s["provenance_auroc"], color="white" if k == "agg3" else c, edgecolor=c, marker=mk, s=46 if ours else 30, linewidths=1.2, zorder=5)
    dx, dy = OFF[k]; ax.annotate(s["label"], (s["quality_partial_rho"], s["provenance_auroc"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.2, ha="left" if dx > 0 else "right", va="center", color="#222222")
ax.set_xlim(-0.45, 0.15); ax.set_ylim(0.4, 1.0); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("tracks reviewer-judged quality\n(partial Spearman vs. ICLR rating, year and domain removed)", fontsize=6.8)
ax.set_ylabel("separates provenance\n(AUROC, SciSlopBench pairs)", fontsize=6.8)
ax.text(-0.44, 0.985, "both", fontsize=6.4, color="#555555", va="top", style="italic")
ax.text(0.14, 0.985, "provenance only", fontsize=6.4, color="#555555", va="top", ha="right", style="italic")
ax.text(-0.44, 0.415, "quality only", fontsize=6.4, color="#555555", va="bottom", style="italic")
ax.annotate("", xy=(-0.42, 0.46), xytext=(-0.28, 0.46), arrowprops=dict(arrowstyle="->", color="#777777", lw=0.6))
ax.text(-0.35, 0.47, "highly rated papers look less AI-like", fontsize=5.4, color="#777777", ha="center")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=7, color="#555555")
fig.tight_layout(pad=0.4)
fig.savefig(f"{HERE}/results/fig_prov_quality_map.pdf"); fig.savefig(f"{HERE}/results/fig_prov_quality_map.png", dpi=300)
for k, s in out["systems"].items(): print(f"{k:11s} quality rho={s['quality_partial_rho']:+.3f} (p {s['quality_p']}) auroc={s['provenance_auroc']}")
