"""Pair metrics for the faithful-setting detector reruns (bench165, 143 pairs).

binoculars_faithful : falcon-7b pair, first 512 tokens -> the PUBLISHED thresholds apply:
                      0.9015 (accuracy-tuned) and 0.8536 (low-fpr), predict AI if B < t.
detectgpt_faithful  : t5-3b, 100 perturbations, GPT-J-6B scorer (surrogate; no published
                      threshold -- the paper reports AUROC only).
Writes results/faithful_summary.json and prints the table.
"""
import json, os, glob, itertools

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/items165.json"))["items"]
label = {i["item_id"]: i["label"] for i in items}
pair_of = {i["item_id"]: i["pair"] for i in items}
pairs = sorted({i["pair"] for i in items})

def load(path, key):
    S = {}
    for f in sorted(glob.glob(path)):
        for l in open(f):
            r = json.loads(l)
            if r.get("id") in label and isinstance(r.get(key), (int, float)):
                S[r["id"]] = r[key]
    return S

def metrics(S, direction):  # direction +1 = AI higher
    corr = n = 0.0
    for p in pairs:
        ai = next((i for i in S if pair_of[i] == p and label[i] == 1), None)
        hu = next((i for i in S if pair_of[i] == p and label[i] == 0), None)
        if ai is None or hu is None: continue
        d = (S[ai] - S[hu]) * direction
        corr += 1.0 if d > 0 else (0.5 if d == 0 else 0.0); n += 1
    A = [S[i]*direction for i in S if label[i] == 1]
    H = [S[i]*direction for i in S if label[i] == 0]
    auroc = sum(1.0 if a > h else (0.5 if a == h else 0.0)
                for a, h in itertools.product(A, H)) / (len(A)*len(H)) if A and H else None
    hs = sorted(H, reverse=True); t5 = hs[int(len(H)*0.05)]
    tpr5 = sum(a > t5 for a in A) / len(A) if A else None
    best = (0, 0, 0)
    for t in sorted(set(A+H)) + [min(A+H)-1]:
        tp = sum(a > t for a in A); fp = sum(h > t for h in H)
        if tp == 0: continue
        pr = tp/(tp+fp); rc = tp/len(A); f1 = 2*pr*rc/(pr+rc)
        if f1 > best[0]: best = (f1, pr, rc)
    return dict(n_pairs=int(n), pairwise=f"{corr}/{int(n)}", pairwise_frac=round(corr/n, 3) if n else None,
                auroc=round(auroc, 3), tpr_at_fpr5=round(tpr5, 3), best_f1=[round(x, 3) for x in best],
                n_ai=len(A), n_hu=len(H))

out = {}
bino = load(f"{B}/results/binoculars_faithful.jsonl", "binoculars")
if bino:
    out["binoculars_faithful"] = metrics(bino, -1)   # lower B = more AI
    for name, t in [("published_acc_0.9015", 0.9015), ("published_lowfpr_0.8536", 0.8536)]:
        A = [v for i, v in bino.items() if label[i] == 1]; H = [v for i, v in bino.items() if label[i] == 0]
        tp = sum(a < t for a in A); fp = sum(h < t for h in H); fn = len(A)-tp
        pr = tp/(tp+fp) if tp+fp else None; rc = tp/len(A)
        out["binoculars_faithful"][name] = dict(
            precision=round(pr, 3) if pr is not None else None, recall=round(rc, 3),
            f1=round(2*pr*rc/(pr+rc), 3) if pr else None, fpr=round(fp/len(H), 3))
dg = load(f"{B}/results/detectgpt_faithful.s*.jsonl", "detectgpt")
if dg:
    out["detectgpt_faithful"] = metrics(dg, +1)      # higher curvature drop = more AI
json.dump(out, open(f"{B}/results/faithful_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
