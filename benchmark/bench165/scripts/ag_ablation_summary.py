"""A3 judge/PMI ablation summary for Argument_Graph on bench165.
Collects every results/slop/argument_graph*/papers.jsonl, computes pair metrics per cell
(slop_score, ABU) and cross-cell Spearman of per-paper scores against the qwen32b cell."""
import json, os, glob, itertools, math
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/items165.json"))["items"]
label = {i["item_id"]: i["label"] for i in items}
pair_of = {i["item_id"]: i["pair"] for i in items}

def load(d):
    S = {}
    p = os.path.join(d, "papers.jsonl")
    if not os.path.exists(p): return S
    for l in open(p):
        r = json.loads(l)
        iid = ("AI_" if r.get("corpus") == "AI" else "HU_") + str(r.get("id"))
        if iid in label:
            S[iid] = {k: r.get(k) for k in ("slop_score", "ABU", "n_key_claims", "status")}
    return S

def metrics(S, key):
    V = {i: v[key] for i, v in S.items() if isinstance(v.get(key), (int, float))}
    pairs = sorted({pair_of[i] for i in V})
    corr = n = 0.0
    for p in pairs:
        ai = next((i for i in V if pair_of[i] == p and label[i] == 1), None)
        hu = next((i for i in V if pair_of[i] == p and label[i] == 0), None)
        if ai is None or hu is None: continue
        d = V[ai] - V[hu]; corr += 1.0 if d > 0 else (0.5 if d == 0 else 0.0); n += 1
    A = [V[i] for i in V if label[i] == 1]; H = [V[i] for i in V if label[i] == 0]
    au = (sum(1.0 if a > h else (0.5 if a == h else 0.0) for a, h in itertools.product(A, H))
          / (len(A) * len(H))) if A and H else None
    return {"pairwise": f"{corr}/{int(n)}", "frac": round(corr / n, 3) if n else None,
            "auroc": round(au, 3) if au else None, "n": len(V)}

def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None

cells = {}
for d in sorted(glob.glob(f"{B}/results/slop/argument_graph*")):
    name = os.path.basename(d).replace("argument_graph", "") or "_qwen32b"
    S = load(d)
    if S:
        cells[name.lstrip("_")] = S
out = {"cells": {}}
base = cells.get("qwen32b")
for name, S in cells.items():
    out["cells"][name] = {k: metrics(S, k) for k in ("slop_score", "ABU")}
    if base and name != "qwen32b":
        common = [i for i in S if i in base
                  and isinstance(S[i].get("slop_score"), (int, float))
                  and isinstance(base[i].get("slop_score"), (int, float))]
        if len(common) > 10:
            out["cells"][name]["spearman_vs_qwen32b"] = {
                "slop_score": round(spearman([S[i]["slop_score"] for i in common],
                                             [base[i]["slop_score"] for i in common]), 3),
                "n_common": len(common)}
json.dump(out, open(f"{B}/results/slop/ag_ablation_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
