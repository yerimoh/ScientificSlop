"""Staircase with baselines: x = FARS / ICLR reject / ICLR accept / ICLR oral, y = mean score percentile
(pooled FARS + ICLR sample per system, oriented AI-like / slop-high). Baselines of Table 2 + SciSlop (4 items).
FARS side: bench165 results (detectors on the same prose view; reviewers from the archived R1 reviews via aggregate165 logic).
ICLR side: results/detectors/*.jsonl (full corpus) and results/reviews/<sys>/ (250-paper subset).
"""
import json, os, glob, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ["FARS", "reject", "accept", "oral"]
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
group = {k: v["group"] for k, v in P.items() if v["group"] in GROUPS}          # FARS ids are FA codes, ICLR ids iclrYYYY_x
items = json.load(open(f"{B}/items165.json"))["items"]
ai_codes = {i["pair"] for i in items if i["label"] == 1}


def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)):
                out[r["id"]] = r[key]
    return out


S, DIR, LABEL = {}, {}, {}
for name, key, d, label in [("binoculars", "binoculars", -1, "Binoculars"), ("detectgpt", "detectgpt", +1, "DetectGPT"), ("nts", "nts", +1, "NTS")]:
    d_f = {k[3:]: v for k, v in jl(f"{B}/results/{key}.jsonl", key).items() if k.startswith("AI_") and k[3:] in ai_codes}
    d_i = jl(f"{HERE}/results/detectors/{key}*.jsonl", key)
    S[name] = {**d_f, **d_i}; DIR[name] = d; LABEL[name] = label
for sysname, dname, label in [("b2h", "B2h_cyclereviewer", "CycleReviewer"), ("b3a", "B3a_ai_scientist", "AI Scientist")]:
    d = {}
    for code in ai_codes:
        fp = f"{ARCH}/{dname}/runs/{code}/logs/{code}_{sysname}_R1_review.json"
        if os.path.exists(fp):
            v = (json.load(open(fp)).get("final") or {}).get("Overall")
            if isinstance(v, (int, float)): d[code] = v
    for f in glob.glob(f"{HERE}/results/reviews/{sysname}/*.json"):
        r = json.load(open(f)); fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)): d[r["item_id"]] = v
    S[f"rev_{sysname}"] = d; DIR[f"rev_{sysname}"] = -1; LABEL[f"rev_{sysname}"] = label
S["scislop_det4"] = {k: v["scislop_det4"] for k, v in P.items() if v.get("scislop_det4") is not None}
DIR["scislop_det4"] = +1; LABEL["scislop_det4"] = "SciSlop (ours)"

out = {}
for name in S:
    ids = [i for i in S[name] if i in group]
    vals = np.array([S[name][i] * DIR[name] for i in ids], float)
    pct = dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))
    g = {gr: [pct[i] for i in ids if group[i] == gr] for gr in GROUPS}
    out[name] = {"label": LABEL[name], "n": {gr: len(g[gr]) for gr in GROUPS},
                 "mean": {gr: (round(float(np.mean(g[gr])), 1) if g[gr] else None) for gr in GROUPS}}
    ordinal = [GROUPS.index(group[i]) for i in ids]
    out[name]["spearman_vs_group"] = round(float(stats.spearmanr(ordinal, [pct[i] for i in ids])[0]), 3)
    icl = [(GROUPS.index(group[i]), pct[i]) for i in ids if group[i] != "FARS"]
    out[name]["spearman_iclr_only"] = round(float(stats.spearmanr([a for a, _ in icl], [b for _, b in icl])[0]), 3)
json.dump({"note": "y = percentile within the pooled FARS+ICLR sample of each system, oriented AI-like / slop-high", "systems": out},
          open(f"{HERE}/results/stairs_systems.json", "w"), indent=1)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
STYLE = {"binoculars": ("#2a78d6", "o"), "detectgpt": ("#1baf7a", "s"), "nts": ("#eda100", "D"),
         "rev_b2h": ("#eb6834", "v"), "rev_b3a": ("#e87ba4", "P"), "scislop_det4": ("#111111", "o")}
fig, ax = plt.subplots(figsize=(3.6, 2.5))
x = list(range(4))
for name in STYLE:
    c, mk = STYLE[name]; ours = name == "scislop_det4"
    y = [out[name]["mean"][gr] for gr in GROUPS]
    ax.plot(x, y, color=c, lw=2.2 if ours else 1.1, marker=mk, ms=5 if ours else 3.2, mec="white" if ours else c, mew=0.8 if ours else 0.5,
            zorder=5 if ours else 3, label=f"{LABEL[name]}   ρ = {out[name]['spearman_vs_group']:+.2f}")
ax.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
ax.set_xticks(x); ax.set_xticklabels(["FARS\n(AI)", "ICLR\nreject", "ICLR\naccept", "ICLR\noral"], fontsize=7.5)
ax.set_ylabel("AI-likeness (percentile of each system's score)", fontsize=7.5)
ax.annotate("more AI-like / more slop", xy=(0.012, 0.985), xycoords="axes fraction", fontsize=6.2, color="#555555", va="top", ha="left", style="italic")
ax.set_ylim(20, 100); ax.set_yticks([20, 40, 60, 80, 100])
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
ax.tick_params(length=2.5, labelsize=7.5, color="#555555"); ax.set_xlim(-0.3, 3.3)
ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=2.2, labelspacing=0.35)
fig.tight_layout(pad=0.4)
fig.savefig(f"{HERE}/results/fig_stairs_systems.pdf"); fig.savefig(f"{HERE}/results/fig_stairs_systems.png", dpi=300)
for name in STYLE:
    o = out[name]; print(f"{name:14s} n={o['n']}  means={o['mean']}  rho_all={o['spearman_vs_group']:+.3f} rho_iclr={o['spearman_iclr_only']:+.3f}")
