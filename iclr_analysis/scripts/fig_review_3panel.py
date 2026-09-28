"""Fig 3 for §5.2 paragraph 3 (0916, replaces fig_review_2panel.py). One row, three panels.
(a) x = mean public review score in four bands, y = each system's score as a within-sample percentile oriented toward AI.
(b) the same percentile (pooled FARS + ICLR) for FARS AI papers and rejected / accepted / oral ICLR papers.
(c) every slop measure that runs on ICLR, mean item score with 95% CI per group (results/stairs.json):
    macro redundancy, cross-section references, argument graph, citation, evidence gap, figure exposition (5 pattern kinds).
SciSlop in (a) and (b) = the paper's aggregation rule (item -> plane -> paper, N/A skipped) over the five measures that
run on ICLR: Structure (macro, xsec), Argument (argument graph, citation), Artifacts (evidence gap). Figure exposition is
observational and appears in (c) only. Data: results/papers_stairs.jsonl (quality-filtered), results/stairs.json,
results/detectors/*.jsonl, results/reviews/<sys>/*.json, bench165 results for the FARS side.
"""
import json, os, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"; ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
ST = json.load(open(f"{HERE}/results/stairs.json"))["main"]
GROUPS = ["FARS", "reject", "accept", "oral"]
group = {k: v["group"] for k, v in P.items() if v["group"] in GROUPS}
rating = {k: v["rating"] for k, v in P.items() if v.get("rating") is not None and v["src"] == "iclr"}
items = json.load(open(f"{B}/items165.json"))["items"]; ai_codes = {i["pair"] for i in items if i["label"] == 1}


def mean_skip(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


AGG = {
    "agg3": lambda v: mean_skip([mean_skip([v.get("macro_redund"), v.get("xsec_ref")]), v.get("citation")]) if None not in (v.get("macro_redund"), v.get("xsec_ref"), v.get("citation")) else None,
    "agg4": lambda v: mean_skip([mean_skip([v.get("macro_redund"), v.get("xsec_ref")]), mean_skip([v.get("argument_graph"), v.get("citation")])]),
    "agg5": lambda v: v.get("scislop5"),
}
AGG_USED = "agg5"


def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)): out[r["id"]] = r[key]
    return out


S, DIR, LABEL = {}, {}, {}
for name, key, d, label in [("binoculars", "binoculars", -1, "Binoculars"), ("detectgpt", "detectgpt", +1, "DetectGPT"), ("nts", "nts", +1, "NTS")]:
    d_f = {k[3:]: v for k, v in jl(f"{B}/results/{key}.jsonl", key).items() if k.startswith("AI_") and k[3:] in ai_codes}
    S[name] = {**d_f, **jl(f"{HERE}/results/detectors/{key}*.jsonl", key)}; DIR[name] = d; LABEL[name] = label
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
for a, fn in AGG.items():
    S[a] = {k: fn(v) for k, v in P.items() if fn(v) is not None}; DIR[a] = +1; LABEL[a] = "SciSlop (ours)"
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", AGG_USED]
STYLE = {"binoculars": ("#2a78d6", "o"), "detectgpt": ("#1baf7a", "s"), "nts": ("#eda100", "D"), "rev_b2h": ("#eb6834", "v"), "rev_b3a": ("#e87ba4", "P"), AGG_USED: ("#111111", "o")}
BANDS = [(2, 3), (4, 5), (6, 7), (8, 9)]


def band_of(r):
    r = int(round(r))
    for k, (a, b) in enumerate(BANDS):
        if a <= r <= b: return k
    return None


def pct(d, ids, sign):
    vals = np.array([d[i] * sign for i in ids], float)
    return dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))


out = {"agg_used": AGG_USED, "left": {}, "right": {}}
left, right = {}, {}
for name in ORDER + [a for a in AGG if a != AGG_USED]:
    ids_l = [i for i in S[name] if i in rating and group.get(i) in ("reject", "accept", "oral")]
    pl = pct(S[name], ids_l, DIR[name])
    rho, p = stats.spearmanr([rating[i] for i in ids_l], [pl[i] for i in ids_l])
    left[name] = [float(np.mean([pl[i] for i in ids_l if band_of(rating[i]) == k])) for k in range(len(BANDS))]
    out["left"][name] = {"n": len(ids_l), "rho_vs_rating": round(float(rho), 3), "p": float(f"{p:.2g}"), "bands": [round(x, 1) for x in left[name]]}
    ids_r = [i for i in S[name] if i in group]
    pr = pct(S[name], ids_r, DIR[name])
    right[name] = [float(np.mean([pr[i] for i in ids_r if group[i] == g])) for g in GROUPS]
    icl = [(GROUPS.index(group[i]), pr[i]) for i in ids_r if group[i] != "FARS"]
    out["right"][name] = {"n": {g: sum(1 for i in ids_r if group[i] == g) for g in GROUPS}, "groups": [round(x, 1) for x in right[name]],
                          "rho_iclr_only": round(float(stats.spearmanr([a for a, _ in icl], [b for _, b in icl])[0]), 3)}
# panel (c): item means with CI from stairs.json
ITEMS = [("macro_redund", "Macro redundancy", "#95627A", "-", "s"), ("xsec_ref", "Cross-section refs.", "#95627A", "--", "^"),
         ("argument_graph", "Argument graph", "#E4959E", "--", "P"), ("citation", "Citation isolation", "#E4959E", "-", "D"),
         ("fig_exposition", "Figure exposition", "#6D8A96", "--", "X"), ("evidence_gap", "Evidence gap", "#6D8A96", "-", "v")]
out["items"] = {k: {g: ST["groups"][k][g] for g in GROUPS} for k, *_ in ITEMS if k in ST["groups"]}
out["items_trend_iclr3"] = {k: ST["trend"][k]["ordinal_iclr3"] for k, *_ in ITEMS if k in ST["trend"]}
json.dump(out, open(f"{HERE}/results/fig_review_3panel.json", "w"), indent=1)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.08], "wspace": 0.34})
XT_G = ["FARS\n(AI)", "ICLR\nreject", "ICLR\naccept", "ICLR\noral"]
for ax, data, xt in [(ax1, left, [f"{a}–{b}" for a, b in BANDS]), (ax2, right, XT_G)]:
    for name in ORDER:
        c, mk = STYLE[name]; ours = name == AGG_USED
        ax.plot(range(4), data[name], color=c, lw=2.2 if ours else 1.0, marker=mk, ms=5 if ours else 3, mec="white" if ours else c, mew=0.8 if ours else 0.5, zorder=5 if ours else 3, label=LABEL[name])
    ax.set_xticks(range(4)); ax.set_xticklabels(xt, fontsize=7.2)
for ax in (ax1, ax2, ax3):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=7.2, color="#555555")
    ax.set_xlim(-0.3, 3.3)
ax1.set_ylim(30, 70); ax1.set_yticks([30, 40, 50, 60, 70]); ax1.set_xlabel("mean review score of ICLR papers", fontsize=7.2)
ax1.set_ylabel("AI-likeness (score percentile)", fontsize=7.2)
ax2.set_ylim(30, 95); ax2.set_yticks([30, 50, 70, 90]); ax2.set_xlabel("paper group", fontsize=7.2)
ax2.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
for k, lab, col, ls, mk in ITEMS:
    if k not in ST["groups"]: continue
    g = ST["groups"][k]; y = [g[gr]["mean"] for gr in GROUPS]
    lo = [y[i] - g[gr]["ci95"][0] for i, gr in enumerate(GROUPS)]; hi = [g[gr]["ci95"][1] - y[i] for i, gr in enumerate(GROUPS)]
    ax3.errorbar(range(4), y, yerr=[lo, hi], color=col, ls=ls, lw=1.0, marker=mk, ms=3.2, mfc="white" if ls == "--" else col, mew=0.7, capsize=1.3, elinewidth=0.5, label=lab, zorder=3)
    NUDGE = {"macro_redund": 5, "fig_exposition": -4, "citation": 3, "argument_graph": -3}     # points, to keep end labels apart
    ax3.annotate(lab, xy=(3, y[3]), xytext=(3, NUDGE.get(k, 0)), textcoords="offset points", fontsize=5.6, color="#333333", va="center", ha="left")
ax3.set_xticks(range(4)); ax3.set_xticklabels(XT_G, fontsize=7.2)
ax3.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
ax3.set_ylim(0, 1.0); ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax3.set_ylabel("slop score (mean, 95% CI)", fontsize=7.2); ax3.set_xlabel("paper group", fontsize=7.2)
for ax, t in [(ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")]:
    ax.text(-0.22, 1.02, t, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.36, -0.02))
fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.30)
fig.savefig(f"{HERE}/results/fig_review_scores3.pdf"); fig.savefig(f"{HERE}/results/fig_review_scores3.png", dpi=300)
for name in ORDER + [a for a in AGG if a != AGG_USED]:
    print(f"{name:12s} left n={out['left'][name]['n']:5d} rho={out['left'][name]['rho_vs_rating']:+.3f} (p {out['left'][name]['p']}) bands={out['left'][name]['bands']} | right={out['right'][name]['groups']} rho_iclr={out['right'][name]['rho_iclr_only']:+.3f}")
for k in out["items"]:
    print(f"  {k:14s} " + " ".join(f"{g}={out['items'][k][g]['mean']}" for g in GROUPS) + f"  iclr3 rho={out['items_trend_iclr3'][k]['spearman']:+.3f} p={out['items_trend_iclr3'][k]['p']}")
