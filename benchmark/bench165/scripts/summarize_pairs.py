"""Pair accuracy + AUROC for one slop papers.jsonl against items165 (session-independent helper).
Usage: summarize_pairs.py <papers.jsonl> <out.json> <key1,key2,...>  (+1 direction: AI higher)"""
import json, sys, itertools, os
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/items165.json"))["items"]
label = {i["item_id"]: i["label"] for i in items}
pair_of = {i["item_id"]: i["pair"] for i in items}
src, out, keys = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
res = {}
for key in keys:
    S = {}
    for l in open(src):
        r = json.loads(l)
        iid = ("AI_" if r.get("corpus") == "AI" else "HU_") + str(r.get("id"))
        v = r.get(key)
        if iid in label and isinstance(v, (int, float)):
            S[iid] = v
    pairs = sorted({pair_of[i] for i in S})
    corr = n = 0.0
    for p in pairs:
        ai = next((i for i in S if pair_of[i] == p and label[i] == 1), None)
        hu = next((i for i in S if pair_of[i] == p and label[i] == 0), None)
        if ai is None or hu is None: continue
        d = S[ai] - S[hu]; corr += 1.0 if d > 0 else (0.5 if d == 0 else 0.0); n += 1
    A = [S[i] for i in S if label[i] == 1]; H = [S[i] for i in S if label[i] == 0]
    au = (sum(1.0 if a > h else (0.5 if a == h else 0.0) for a, h in itertools.product(A, H))
          / (len(A) * len(H))) if A and H else None
    res[key] = {"pairwise": f"{corr}/{int(n)}", "pairwise_frac": round(corr / n, 3) if n else None,
                "auroc": round(au, 3) if au else None, "n_ai": len(A), "n_hu": len(H)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
