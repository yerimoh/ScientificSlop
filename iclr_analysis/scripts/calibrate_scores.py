r"""Put every system on one interpretable axis, calibrated AI probability, so the systems can be compared to each
other and not only to themselves.

The problem with a percentile. Ranking each system inside the corpus makes every system uniform on the same range,
so a detector that barely separates anything and a measure that separates strongly draw lines of the same height.
The differences between the baselines disappear.

The fix. Each system is calibrated once on SciSlopBench, where the answer is known, by fitting a one dimensional
logistic regression of the label (AI or human) on that system's raw score over the 143 AI papers and their 143
human counterparts. The fitted curve turns any raw score into the probability that system assigns to AI authorship.
Applied to the ICLR papers it gives one axis in the same units for every system, and a system that cannot separate
the two classes produces probabilities that sit near the benchmark base rate of one half with almost no spread,
which is exactly the difference a percentile hides.

Outputs results/calibration.json (fitted coefficients, benchmark AUROC, the spread each system produces on ICLR)
and adds <sys>_pai columns to results/scores_years_norm.csv and results/scores_2026_norm.csv.
"""
import csv, glob, json, os
import numpy as np
from scipy import stats, optimize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
SYS = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3", "agg4", "macro_redund", "xsec_ref", "citation"]


def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r.get(key), (int, float)):
                out[r["id"]] = r[key]
    return out


def bench_scores():
    """id -> (raw score, label) for every system on the 143 benchmark pairs."""
    items = json.load(open(f"{B}/items165.json"))["items"]
    label = {i["item_id"]: i["label"] for i in items}
    out = {s: {"ai": [], "hu": []} for s in SYS}
    for s, key in (("binoculars", "binoculars"), ("detectgpt", "detectgpt"), ("nts", "nts")):
        for iid, v in jl(f"{B}/results/{key}.jsonl", key).items():
            if iid in label:
                out[s]["ai" if label[iid] == 1 else "hu"].append(v)
    for s, sysname, dname in (("rev_b2h", "b2h", "B2h_cyclereviewer"), ("rev_b3a", "b3a", "B3a_ai_scientist")):
        for f in glob.glob(f"{B}/results/reviews/{sysname}/*.json"):
            r = json.load(open(f)); fin = r.get("final") or {}
            v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
            if isinstance(v, (int, float)) and r.get("item_id") in label:
                out[s]["ai" if label[r["item_id"]] == 1 else "hu"].append(v)
        for f in glob.glob(f"{ARCH}/{dname}/runs/FA*/logs/*_{sysname}_R1_review.json"):
            v = ((json.load(open(f)) or {}).get("final") or {}).get("Overall")
            if isinstance(v, (int, float)):
                out[s]["ai"].append(v)
    rows = {}
    for item in ("macro_redund", "xsec_ref", "citation", "evidence_gap"):
        p = f"{B}/results/slop/{item}/papers.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            v = r.get("slop_score_agg") if item == "macro_redund" else r.get("slop_score")
            rows.setdefault(r["id"], {"corpus": r["corpus"]})[item] = v
    for pid, v in rows.items():
        st = [v.get("macro_redund"), v.get("xsec_ref")]
        side = "ai" if v["corpus"] == "AI" else "hu"
        for s in ("macro_redund", "xsec_ref", "citation"):
            if v.get(s) is not None:
                out[s][side].append(v[s])
        if None not in (st[0], st[1], v.get("citation")):
            a3 = ((st[0] + st[1]) / 2 + v["citation"]) / 2
            out["agg3"][side].append(a3)
            planes = [(st[0] + st[1]) / 2, v["citation"]] + ([v["evidence_gap"]] if v.get("evidence_gap") is not None else [])
            out["agg4"][side].append(float(np.mean(planes)))
    return out


def fit_logistic(ai, hu):
    """One dimensional logistic regression of the AI label on the raw score, standardised for conditioning."""
    x = np.array(ai + hu, float); y = np.array([1.0] * len(ai) + [0.0] * len(hu))
    mu, sd = float(np.mean(x)), float(np.std(x)) or 1.0
    z = (x - mu) / sd

    def nll(p):
        a, b = p
        t = a + b * z
        return float(np.sum(np.logaddexp(0, t) - y * t)) + 1e-3 * (a * a + b * b)

    r = optimize.minimize(nll, [0.0, 1.0], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
    a, b = map(float, r.x)
    pos = np.array(ai, float); neg = np.array(hu, float)
    raw = float(np.mean([(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]))
    # the raw value is below one half when the AI-like end of the scale is the low one; the fitted slope carries the
    # direction, so the separation is reported after orienting it
    return {"a": a, "b": b, "mu": mu, "sd": sd, "n_ai": len(ai), "n_hu": len(hu),
            "auroc_raw": round(raw, 3), "auroc": round(max(raw, 1 - raw), 3), "ai_end": "high" if raw >= 0.5 else "low"}


def apply_cal(cal, v):
    if v is None or cal is None:
        return None
    z = (v - cal["mu"]) / cal["sd"]
    return float(1.0 / (1.0 + np.exp(-(cal["a"] + cal["b"] * z))))


bs = bench_scores()
CAL = {}
for s in SYS:
    if len(bs[s]["ai"]) >= 30 and len(bs[s]["hu"]) >= 30:
        CAL[s] = fit_logistic(bs[s]["ai"], bs[s]["hu"])
OUT = {"note": __doc__, "calibration": CAL, "applied": {}}

for path in (f"{HERE}/results/scores_years_norm.csv", f"{HERE}/results/scores_2026_norm.csv"):
    if not os.path.exists(path):
        continue
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0].keys())
    for s in CAL:
        col = f"{s}_pai"
        if col not in fields:
            fields.append(col)
        vals = []
        for r in rows:
            raw = r.get(s)
            v = None if raw in (None, "", "None") else float(raw)
            p = apply_cal(CAL[s], v)
            r[col] = "" if p is None else round(p, 4)
            if p is not None:
                vals.append(p)
        if vals:
            OUT["applied"].setdefault(os.path.basename(path), {})[s] = {
                "n": len(vals), "mean": round(float(np.mean(vals)), 3),
                "p10": round(float(np.percentile(vals, 10)), 3), "p90": round(float(np.percentile(vals, 90)), 3),
                "spread_p10_p90": round(float(np.percentile(vals, 90) - np.percentile(vals, 10)), 3)}
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    os.replace(tmp, path)
json.dump(OUT, open(f"{HERE}/results/calibration.json", "w"), indent=1)

print("calibrated on the 143 SciSlopBench pairs, one logistic per system")
print(f"{'system':14s} {'bench AUROC':>10s} {'AI n':>6s} {'human n':>7s} {'ICLR mean P(AI)':>16s} {'10-90% width':>11s}")
for s, c in CAL.items():
    ap = (OUT["applied"].get("scores_years_norm.csv") or {}).get(s, {})
    print(f"{s:14s} {c['auroc']:>10.3f} {c['n_ai']:>6d} {c['n_hu']:>7d} {ap.get('mean', float('nan')):>16.3f} {ap.get('spread_p10_p90', float('nan')):>11.3f}")
