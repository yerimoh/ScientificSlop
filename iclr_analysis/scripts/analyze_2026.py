r"""Single-year analysis on ICLR 2026 (Pangram release + arXiv tex).

Why this corpus. The 2017-2026 corpus mixes review scales and its score bands mix years, so a raw mean rating is not
comparable across papers. ICLR 2026 is one year and one scale, and it carries what no other year has: the review
sub-scores (soundness, presentation, contribution, confidence) and Pangram's own AI judgment, the detector ICLR used.

Cleaning (fixed before looking at any result):
  extraction    the audit rules of scripts/quality_audit.py (missing \input, section parse failure, < 3 body
                sections, < 5 objects, prose view < 1,500 words, any item below its minimum denominator)
  reviews       at least 3 reviews, so one reviewer cannot set a paper's score
  length        body length inside the 1st to 99th percentile of the corpus
Outputs results/analysis_2026.json and prints every number.
"""
import json, os, glob, collections, itertools, math
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SCISLOP_ROOT", "."); B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
rng = np.random.default_rng(0)
ITEMS = ["macro_redund", "xsec_ref", "citation", "evidence_gap"]
LAB = {"rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "macro_redund": "Macro redundancy", "xsec_ref": "Cross-section refs.", "citation": "Citation isolation",
       "evidence_gap": "Evidence gap", "agg3": "SciSlop (3 items)", "agg4": "SciSlop (4 items)",
       "pangram": "Pangram (fraction AI)", "binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS"}

recs = {r["id"]: r for r in json.load(open(f"{HERE}/records/iclr2026_records.json"))["records"]}
rows = {it: {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/slop/{it}_2026/papers.jsonl")} for it in ITEMS}
words = {json.loads(l)["id"]: json.loads(l)["body_words"] for l in open(f"{HERE}/records/texts_iclr2026.jsonl")}

# ---------------------------------------------------------------- cleaning
flags = collections.defaultdict(set)
for pid, r in recs.items():
    m, x, c = rows["macro_redund"].get(pid), rows["xsec_ref"].get(pid), rows["citation"].get(pid)
    if m and m.get("source_incomplete"): flags["tex incomplete"].add(pid)
    if c and c.get("status") == "no_sections": flags["section parse failure"].add(pid)
    if m and (m.get("n_sections") or 0) < 3: flags["< 3 sections"].add(pid)
    if x and (x.get("n_sections_body") or 0) < 3: flags["< 3 body sections"].add(pid)
    if x and (x.get("n_objects") or 0) < 5: flags["< 5 objects"].add(pid)
    if words.get(pid, 0) < 1500: flags["prose < 1500 words"].add(pid)
    for k, d in (("macro", m), ("xsec", x), ("citation", c)):
        if d and d.get("weak"): flags[f"{k} below minimum denominator"].add(pid)
    if (r.get("n_reviews") or 0) < 3: flags["< 3 reviews"].add(pid)
w = np.array([words.get(p, 0) for p in recs])
lo, hi = np.percentile(w[w > 0], [1, 99])
for pid in recs:
    if words.get(pid, 0) and not (lo <= words[pid] <= hi): flags["length outside 1-99 pct"].add(pid)
excl = set().union(*flags.values()) if flags else set()
keep = [p for p in recs if p not in excl]

P = {}
for pid in keep:
    r = recs[pid]
    v = {"id": pid, "rating": r["overall_mean"], "soundness": r["soundness"], "presentation": r["presentation"],
         "contribution": r["contribution"], "confidence": r["confidence"], "accept": r["accept"], "group": r["group4"],
         "fraction_ai": r["fraction_ai"], "n_reviews": r["n_reviews"], "words": words.get(pid),
         "spread": (r["rating_max"] - r["rating_min"]) if r.get("rating_max") is not None else None,
         "area": r.get("primary_area")}
    for it in ITEMS:
        d = rows[it].get(pid) or {}
        v[it] = d.get("slop_score_agg") if it == "macro_redund" else d.get("slop_score")
    st = [v["macro_redund"], v["xsec_ref"]]
    v["agg3"] = ((st[0] + st[1]) / 2 + v["citation"]) / 2 if None not in (st[0], st[1], v["citation"]) else None
    v["agg4"] = None if v["agg3"] is None else (v["agg3"] * 2 + (v["evidence_gap"] if v["evidence_gap"] is not None else 0)) / (3 if v["evidence_gap"] is not None else 2) if False else None
    if None not in (st[0], st[1], v["citation"]):
        planes = [(st[0] + st[1]) / 2, v["citation"]] + ([v["evidence_gap"]] if v["evidence_gap"] is not None else [])
        v["agg4"] = float(np.mean(planes))
    P[pid] = v

# detector scores (GPU jobs may still be running; whatever is on disk is used)
def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)): out[r["id"]] = r[key]
    return out
DET = {"binoculars": (jl(f"{HERE}/results/detectors/binoculars.p26*.jsonl", "binoculars"), -1),
       "detectgpt": (jl(f"{HERE}/results/detectors/detectgpt.p26*.jsonl", "detectgpt"), +1),
       "nts": (jl(f"{HERE}/results/detectors/nts.p26*.jsonl", "nts"), +1)}
# the two reviewer models, scored on the stratified subset only; Overall is a 1-10 rating, so lower means more AI-like
for name, sub in (("rev_b2h", "b2h_2026"), ("rev_b3a", "b3a_2026")):
    d = {}
    for f in glob.glob(f"{HERE}/results/reviews/{sub}/*.json"):
        try:
            r = json.load(open(f)); fin = r.get("final") or {}
        except Exception:
            continue
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)):
            d[os.path.splitext(os.path.basename(f))[0]] = v
    DET[name] = (d, -1)
for name, (d, sign) in DET.items():
    for pid, v in P.items():
        if pid in d: v[name] = d[pid] * sign
for v in P.values():
    v["pangram"] = v["fraction_ai"]
SYS = ["pangram", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3", "agg4"] + ITEMS

def spear(xs, ys, boot=2000):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 10 or len(set(xs)) < 3 or len(set(ys)) < 3: return None
    rho, p = stats.spearmanr(xs, ys)
    bs = []
    for i in rng.integers(0, len(xs), (boot, len(xs))):
        if len(set(xs[i])) > 1 and len(set(ys[i])) > 1: bs.append(stats.spearmanr(xs[i], ys[i])[0])
    return {"n": int(len(xs)), "rho": round(float(rho), 3), "p": float(f"{p:.3g}"),
            "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}

def pairs(key, sel=None, target="rating"):
    sel = sel if sel is not None else P.values()
    return [(v[key], v[target]) for v in sel if v.get(key) is not None and v.get(target) is not None]

OUT = {"corpus": "ICLR 2026, Pangram release + arXiv tex, one year and one review scale",
       "n_records": len(recs), "excluded": {k: len(v) for k, v in flags.items()}, "n_excluded": len(excl), "n_kept": len(keep),
       "note": "every score oriented so that higher means more AI-like; rho against the mean public review score"}

# ---------------------------------------------------------------- 1 rating correlation
OUT["rating"] = {}
for k in SYS:
    xy = pairs(k)
    r = spear([a for a, _ in xy], [b for _, b in xy]) if len(xy) >= 30 else None
    if r: OUT["rating"][k] = r
# ---------------------------------------------------------------- 2 review dimensions
OUT["dimensions"] = {}
for k in SYS:
    OUT["dimensions"][k] = {}
    for dim in ["rating", "soundness", "presentation", "contribution", "confidence"]:
        xy = pairs(k, target=dim)
        r = spear([a for a, _ in xy], [b for _, b in xy], boot=600) if len(xy) >= 30 else None
        if r: OUT["dimensions"][k][dim] = r
# partial: soundness controlling presentation and vice versa
def partial(k, a, b):
    sel = [v for v in P.values() if v.get(k) is not None and v.get(a) is not None and v.get(b) is not None]
    if len(sel) < 30: return None
    X = stats.rankdata([v[k] for v in sel]); A = stats.rankdata([v[a] for v in sel]); Bv = stats.rankdata([v[b] for v in sel])
    def resid(u, z):
        Z = np.vstack([z, np.ones_like(z)]).T
        return u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
    r, p = stats.pearsonr(resid(X, Bv), resid(A, Bv))
    return {"n": len(sel), "partial_rho": round(float(r), 3), "p": float(f"{p:.3g}")}
OUT["dimensions_partial"] = {k: {"soundness_given_presentation": partial(k, "soundness", "presentation"),
                                 "presentation_given_soundness": partial(k, "presentation", "soundness")} for k in SYS}
# ---------------------------------------------------------------- 3 Pangram-clean layer
clean = [v for v in P.values() if v.get("fraction_ai") == 0]
lowai = [v for v in P.values() if (v.get("fraction_ai") or 0) <= 0.05]
OUT["pangram_clean_layer"] = {"n_fraction_ai_0": len(clean), "n_le_0.05": len(lowai), "systems": {}}
for k in SYS:
    for tag, sel in (("fraction_ai == 0", clean), ("fraction_ai <= 0.05", lowai)):
        xy = pairs(k, sel)
        r = spear([a for a, _ in xy], [b for _, b in xy], boot=600) if len(xy) >= 30 else None
        if r: OUT["pangram_clean_layer"]["systems"].setdefault(k, {})[tag] = r
# ---------------------------------------------------------------- 4 accept, and accept at matched rating
def auroc(pos, neg):
    if not pos or not neg: return None
    return round(float(sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))), 3)
OUT["accept"] = {}
for k in SYS:
    rej = [v[k] for v in P.values() if v.get(k) is not None and not v["accept"]]
    acc = [v[k] for v in P.values() if v.get(k) is not None and v["accept"]]
    if len(rej) >= 20 and len(acc) >= 20:
        OUT["accept"][k] = {"n_reject": len(rej), "n_accept": len(acc), "auroc_reject_over_accept": auroc(rej, acc),
                            "mwu_p": float(f"{stats.mannwhitneyu(rej, acc).pvalue:.3g}")}
# ---------------------------------------------------------------- 5 incremental validity
def partial_vs_rating(k, ctrl):
    sel = [v for v in P.values() if v.get(k) is not None and v.get(ctrl) is not None]
    if len(sel) < 50: return None
    X = stats.rankdata([v[k] for v in sel]); Y = stats.rankdata([v["rating"] for v in sel]); Z = stats.rankdata([v[ctrl] for v in sel])
    def resid(u):
        Zm = np.vstack([Z, np.ones_like(Z)]).T
        return u - Zm @ np.linalg.lstsq(Zm, u, rcond=None)[0]
    r, p = stats.pearsonr(resid(X), resid(Y))
    return {"n": len(sel), "partial_rho": round(float(r), 3), "p": float(f"{p:.3g}")}
OUT["incremental"] = {k: {c: partial_vs_rating(k, c) for c in ["pangram", "binoculars", "detectgpt", "nts", "words"] if c != k} for k in ["agg3", "agg4", "pangram"]}
OUT["incremental"]["agg3_given_length"] = partial_vs_rating("agg3", "words")
# ---------------------------------------------------------------- 6 how the systems relate to each other
OUT["cross_system"] = {}
for a, b in itertools.combinations(["pangram", "binoculars", "detectgpt", "nts", "agg3"], 2):
    sel = [v for v in P.values() if v.get(a) is not None and v.get(b) is not None]
    r = spear([v[a] for v in sel], [v[b] for v in sel], boot=400) if len(sel) >= 50 else None
    if r: OUT["cross_system"][f"{a} x {b}"] = r
# ---------------------------------------------------------------- 7 reviewer disagreement
OUT["disagreement"] = {}
for k in ["agg3", "pangram"] + ITEMS:
    xy = pairs(k, target="spread")
    r = spear([a for a, _ in xy], [b for _, b in xy], boot=600) if len(xy) >= 50 else None
    if r: OUT["disagreement"][k] = r
json.dump(OUT, open(f"{HERE}/results/analysis_2026.json", "w"), indent=1)
with open(f"{HERE}/results/papers_2026.jsonl", "w") as fh:
    for v in P.values(): fh.write(json.dumps(v) + "\n")

# ---------------------------------------------------------------- print
print(f"records {len(recs)} -> kept {len(keep)} (excluded {len(excl)})")
print("  exclusions:", {k: len(v) for k, v in sorted(flags.items(), key=lambda kv: -len(kv[1]))})
print("\n1. correlation with the mean review score (higher score = more AI-like)")
for k, v in OUT["rating"].items():
    print(f"  {LAB.get(k,k):22s} n={v['n']:4d} rho={v['rho']:+.3f} p={v['p']:<9.3g} ci={v['ci95']}")
print("\n2. which review dimension does each system track")
hdr = ["rating", "soundness", "presentation", "contribution", "confidence"]
print(f"  {'system':22s} " + " ".join(f"{h[:9]:>10s}" for h in hdr))
for k, d in OUT["dimensions"].items():
    if not d: continue
    print(f"  {LAB.get(k,k):22s} " + " ".join(f"{d[h]['rho']:+10.3f}" if h in d else f"{'-':>10s}" for h in hdr))
print("\n  partial (soundness | presentation) and (presentation | soundness)")
for k, d in OUT["dimensions_partial"].items():
    a, b = d["soundness_given_presentation"], d["presentation_given_soundness"]
    if a and b: print(f"  {LAB.get(k,k):22s} sound|pres {a['partial_rho']:+.3f} (p={a['p']:.2g})   pres|sound {b['partial_rho']:+.3f} (p={b['p']:.2g})")
print(f"\n3. inside the layer Pangram calls human (fraction_ai == 0, n={len(clean)}; <= 0.05, n={len(lowai)})")
for k, d in OUT["pangram_clean_layer"]["systems"].items():
    for tag, v in d.items():
        print(f"  {LAB.get(k,k):22s} {tag:20s} n={v['n']:4d} rho={v['rho']:+.3f} p={v['p']:.3g}")
print("\n4. rejected vs accepted (AUROC of reject over accept)")
for k, v in OUT["accept"].items():
    print(f"  {LAB.get(k,k):22s} auroc={v['auroc_reject_over_accept']:.3f} p={v['mwu_p']:.3g} (n {v['n_reject']}/{v['n_accept']})")
print("\n5. incremental validity, partial rho against rating after removing another system")
for k, d in OUT["incremental"].items():
    if not isinstance(d, dict) or "partial_rho" in d: continue
    for c, v in d.items():
        if v: print(f"  {LAB.get(k,k):22s} given {LAB.get(c,c):22s} rho={v['partial_rho']:+.3f} p={v['p']:.3g}")
print("\n6. agreement between systems")
for k, v in OUT["cross_system"].items(): print(f"  {k:28s} rho={v['rho']:+.3f} p={v['p']:.3g}")
print("\n7. reviewer disagreement (rating max minus min)")
for k, v in OUT["disagreement"].items(): print(f"  {LAB.get(k,k):22s} rho={v['rho']:+.3f} p={v['p']:.3g}")

# ---------------------------------------------------------------- 8 design-aware estimates
# The 2026 texts were collected stratified by Pangram band x decision (ICLR2026_Pangram/collect_texts.py), so a raw
# association between a Pangram-correlated variable and the decision inside this sample is a sampling artifact.
# Everything below stays inside one stratum and then pools, which the design cannot distort.
def fa_band(v):
    f = v.get("fraction_ai")
    if f is None: return None
    return "0" if f == 0 else ("0-0.1" if f <= 0.1 else ("0.1-0.5" if f <= 0.5 else "0.5+"))
for v in P.values(): v["fa_band"] = fa_band(v)
def within(key, by, target="rating", min_cell=25):
    cells = collections.defaultdict(list)
    for v in P.values():
        if v.get(key) is not None and v.get(target) is not None and v.get(by) is not None: cells[v[by]].append(v)
    per, ws = {}, []
    for c, vs in sorted(cells.items()):
        if len(vs) < min_cell: continue
        rho = stats.spearmanr([v[key] for v in vs], [v[target] for v in vs])[0]
        if rho == rho:
            per[str(c)] = {"n": len(vs), "rho": round(float(rho), 3)}; ws.append((len(vs), rho))
    if not ws: return None
    pooled = sum(n * r for n, r in ws) / sum(n for n, _ in ws)
    return {"pooled_rho": round(float(pooled), 3), "n_cells": len(ws), "n": sum(n for n, _ in ws), "per_cell": per}
OUT["within_strata"] = {}
for k in SYS:
    OUT["within_strata"][k] = {"by_pangram_band": within(k, "fa_band"), "by_decision": within(k, "accept"),
                               "by_area": within(k, "area", min_cell=40)}
# rating correlation inside accepted papers only and inside rejected papers only
OUT["by_decision_detail"] = {}
for k in ["agg3", "agg4", "citation", "pangram"] + [d for d in ("binoculars", "detectgpt", "nts")]:
    for tag, sel in (("accepted", [v for v in P.values() if v["accept"]]), ("rejected", [v for v in P.values() if not v["accept"]])):
        xy = [(v[k], v["rating"]) for v in sel if v.get(k) is not None]
        r = spear([a for a, _ in xy], [b for _, b in xy], boot=600) if len(xy) >= 30 else None
        if r: OUT["by_decision_detail"].setdefault(k, {})[tag] = r
# ---------------------------------------------------------------- 9 provenance x quality, per item
bench = {}
for it in ITEMS + ["argument_graph", "fig_graph"]:
    f = f"{B}/results/slop/{it}/papers.jsonl"
    if not os.path.exists(f): continue
    rr = [json.loads(l) for l in open(f)]
    key = "slop_score"
    ai = [r[key] for r in rr if r["corpus"] == "AI" and r.get(key) is not None]
    hu = [r[key] for r in rr if r["corpus"] == "HU" and r.get(key) is not None]
    if ai and hu: bench[it] = {"auroc": auroc(ai, hu), "n_ai": len(ai), "n_hu": len(hu)}
OUT["provenance_vs_quality_items"] = {}
for it in ITEMS:
    q = OUT["rating"].get(it); qc = (OUT["pangram_clean_layer"]["systems"].get(it) or {}).get("fraction_ai == 0")
    OUT["provenance_vs_quality_items"][it] = {"label": LAB[it], "provenance_auroc": (bench.get(it) or {}).get("auroc"),
                                              "quality_rho_2026": q["rho"] if q else None,
                                              "quality_rho_pangram_clean": qc["rho"] if qc else None}
json.dump(OUT, open(f"{HERE}/results/analysis_2026.json", "w"), indent=1)
print("\n8. design-aware, correlation with rating pooled inside strata")
for k, d in OUT["within_strata"].items():
    b, c = d["by_pangram_band"], d["by_decision"]
    if b or c:
        print(f"  {LAB.get(k,k):22s} within Pangram band {(b or {}).get('pooled_rho')}  within decision {(c or {}).get('pooled_rho')}  within area {(d['by_area'] or {}).get('pooled_rho')}")
print("\n  inside accepted / inside rejected")
for k, d in OUT["by_decision_detail"].items():
    for tag, v in d.items(): print(f"  {LAB.get(k,k):22s} {tag:9s} n={v['n']:4d} rho={v['rho']:+.3f} p={v['p']:.3g}")
print("\n9. per item, provenance separation (bench165 AUROC) against quality tracking (2026 rho)")
for it, v in OUT["provenance_vs_quality_items"].items():
    print(f"  {v['label']:22s} AUROC={v['provenance_auroc']}  rho(rating)={v['quality_rho_2026']}  rho inside Pangram-clean={v['quality_rho_pangram_clean']}")

# ---------------------------------------------------------------- 10 the same papers for every system
# The detectors run on the whole corpus and the reviewers only on a stratified subset, so a correlation computed on
# each system's own coverage is not a like-for-like comparison. This block repeats the headline numbers on the
# papers where every system has a score.
CORE = ["pangram", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]
common = [v for v in P.values() if all(v.get(k) is not None for k in CORE)]
OUT["common_subset"] = {"n": len(common), "systems": CORE, "rating": {}, "note":
                        "every system scored on exactly these papers; the wider per-system numbers are in OUT['rating']"}
for k in CORE + ["agg4", "citation", "macro_redund", "xsec_ref"]:
    xy = [(v[k], v["rating"]) for v in common if v.get(k) is not None]
    r = spear([a for a, _ in xy], [b for _, b in xy], boot=1000) if len(xy) >= 30 else None
    if r: OUT["common_subset"]["rating"][k] = r
if common:
    # every half-point review score that holds at least MIN_CELL of the common papers, so the axis keeps the
    # intermediate scores instead of collapsing them. Mean percentile with its standard error, not a threshold share.
    MIN_CELL = 40
    OUT["common_subset"]["levels"] = {}
    OUT["common_subset"]["min_cell"] = MIN_CELL
    for k in CORE:
        vals = [v for v in common if v.get(k) is not None]
        arr = np.array([v[k] for v in vals], float)
        pct = dict(zip([v["id"] for v in vals], stats.rankdata(arr) / len(arr) * 100))
        cells = {}
        for v in vals:
            cells.setdefault(round(v["rating"] * 2) / 2, []).append(pct[v["id"]])
        OUT["common_subset"]["levels"][k] = {
            str(b): {"n": len(ps), "mean_percentile": round(float(np.mean(ps)), 1),
                     "sem": round(float(np.std(ps, ddof=1) / np.sqrt(len(ps))), 1) if len(ps) > 1 else None}
            for b, ps in sorted(cells.items()) if len(ps) >= MIN_CELL}
json.dump(OUT, open(f"{HERE}/results/analysis_2026.json", "w"), indent=1)
print(f"\n10. same-paper comparison, n = {len(common)}")
for k, v in OUT["common_subset"]["rating"].items():
    print(f"  {LAB.get(k,k):22s} rho={v['rho']:+.3f} p={v['p']:.3g} ci={v['ci95']}")
