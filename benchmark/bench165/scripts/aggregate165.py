"""Aggregate bench165 columns; pair metrics overall and stratified by sim_tier>=2 and pool.
Also M5: Spearman between each system's HU-side score and the true ICLR rating."""
import json, os, csv, glob, itertools

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/items165.json"))["items"]
ids = [i["item_id"] for i in items]
meta = {i["item_id"]: i for i in items}
pairs = sorted({i["pair"] for i in items})
S = {i: {} for i in ids}

for it in items:
    S[it["item_id"]]["triv_body_words"] = it["body_words"]

for name, path, keys in [("binoculars", f"{B}/results/binoculars.jsonl", ("binoculars",)),
                         ("detectgpt", f"{B}/results/detectgpt.jsonl", ("detectgpt",))]:
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            v = next((r[k] for k in keys if isinstance(r.get(k), (int, float))), None)
            if r.get("id") in S and v is not None:
                S[r["id"]][name] = v

for sysname in ["b3a", "b2h", "b3i"]:
    for f in glob.glob(f"{B}/results/reviews/{sysname}/*.json"):
        r = json.load(open(f))
        fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if r.get("item_id") in S and isinstance(v, (int, float)):
            S[r["item_id"]][f"rev_{sysname}"] = v
# FARS side from the 165-run archive (R1 reviews of the original papers)
ARCH = os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/Evaluation/02_baselines_B"
for sysname, d in [("b3a", "B3a_ai_scientist"), ("b2h", "B2h_cyclereviewer")]:
    for it in items:
        if it["label"] != 1:
            continue
        fp = f"{ARCH}/{d}/runs/{it['pair']}/logs/{it['pair']}_{sysname}_R1_review.json"
        if os.path.exists(fp):
            fin = json.load(open(fp)).get("final") or {}
            v = fin.get("Overall")
            if isinstance(v, (int, float)):
                S[it["item_id"]][f"rev_{sysname}"] = v

def slop_id(rid):
    return f"AI_{rid}" if rid.startswith("FA") else f"HU_{rid}"
for name in ["macro_redund", "xsec_ref", "claim_table", "citation"]:
    p = f"{B}/results/slop/{name}/papers.jsonl"
    if not os.path.exists(p):
        continue
    for l in open(p):
        r = json.loads(l)
        rid = slop_id(r["id"])
        if rid in S and r.get("slop_score") is not None:
            S[rid][f"slop_{name}"] = r["slop_score"]

DIRECTION = {"binoculars": -1, "detectgpt": +1,
             "rev_b3a": -1, "rev_b2h": -1, "rev_b3i": -1,
             "slop_macro_redund": +1, "slop_xsec_ref": +1, "slop_claim_table": +1, "slop_citation": +1,
             "triv_body_words": -1}

def evaluate(subset_pairs, col):
    corr = n = 0.0
    for p in subset_pairs:
        ai = next(i for i in ids if meta[i]["pair"] == p and meta[i]["label"] == 1)
        hu = next(i for i in ids if meta[i]["pair"] == p and meta[i]["label"] == 0)
        va, vh = S[ai].get(col), S[hu].get(col)
        if va is None or vh is None:
            continue
        d = DIRECTION[col]
        corr += 1.0 if (va - vh) * d > 0 else (0.5 if va == vh else 0.0); n += 1
    A = [S[i][col] for i in ids if meta[i]["label"] == 1 and col in S[i] and meta[i]["pair"] in subset_pairs]
    H = [S[i][col] for i in ids if meta[i]["label"] == 0 and col in S[i] and meta[i]["pair"] in subset_pairs]
    auroc = None
    if A and H:
        d = DIRECTION[col]
        auroc = sum(1.0 if (a - h) * d > 0 else (0.5 if a == h else 0.0)
                    for a, h in itertools.product(A, H)) / (len(A) * len(H))
    return (corr, n), auroc

def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda k: v[k]); r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None

tier2 = {i["pair"] for i in items if (i.get("sim_tier") or 0) >= 2}
strata = {"all": set(pairs), "tier>=2": tier2, "cited": {i["pair"] for i in items if i["pool"] == "cited"},
          "widened": {i["pair"] for i in items if i["pool"] == "widened_iclr26"}}
out = {}
cols = sorted({k for i in ids for k in S[i]} & set(DIRECTION))
for col in cols:
    out[col] = {}
    for sname, sp in strata.items():
        (c, n), au = evaluate(sp, col)
        out[col][sname] = {"pairwise": f"{c}/{int(n)}", "auroc": round(au, 3) if au else None}
# M5: HU-side score vs true rating
m5 = {}
hu_rated = [i for i in ids if meta[i]["label"] == 0 and isinstance(meta[i].get("iclr_rating"), (int, float))]
for col in cols:
    xs = [(S[i][col], meta[i]["iclr_rating"]) for i in hu_rated if col in S[i]]
    if len(xs) >= 10:
        m5[col] = {"n": len(xs), "spearman": round(spearman([a for a, _ in xs], [b for _, b in xs]), 3)}
json.dump({"pair_metrics": out, "m5_vs_true_rating": m5}, open(f"{B}/results/pairwise165.json", "w"), indent=1)

with open(f"{B}/results/scores165.csv", "w", newline="") as fh:
    allc = sorted({k for i in ids for k in S[i]})
    w = csv.writer(fh); w.writerow(["item_id", "pair", "label", "sim_tier", "pool"] + allc)
    for i in ids:
        w.writerow([i, meta[i]["pair"], meta[i]["label"], meta[i].get("sim_tier"), meta[i].get("pool")] + [S[i].get(c, "") for c in allc])

print(f"{'column':22s} {'all':>12s} {'tier>=2':>12s} {'cited':>12s} {'widened':>12s}")
for col in cols:
    row = out[col]
    print(f"{col:22s} " + " ".join(f"{row[s]['pairwise']:>7s}({row[s]['auroc']})" if row[s]['auroc'] is not None else f"{row[s]['pairwise']:>7s}(--)" for s in ['all','tier>=2','cited','widened']))
print("\nM5 (HU score vs true ICLR rating):", json.dumps(m5))
