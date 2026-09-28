"""DetectGPT (Mitchell et al., ICML 2023) on the ICLR 2017-2025 corpus.

d(x) = [ log p(x) - mean_i log p(x~_i) ] / std_i log p(x~_i)
x~ are T5 mask-filling perturbations of x (2-word spans, 15% of the text).
Higher d = more machine-like (machine text sits at negative curvature).

Deviation from the paper, stated so it can be checked: n_perturb defaults to 20
rather than 100, and the perturber is t5-large rather than t5-3b, to fit 1,294
full papers into the compute budget. Both reduce power, not validity: the estimator
is unbiased in n_perturb, only noisier.
"""
import os, re, json, argparse, time, random
import numpy as np, torch, torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          T5TokenizerFast, T5ForConditionalGeneration)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--scorer", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--perturber", default="t5-large")
ap.add_argument("--texts", default=os.path.join(HERE, "texts.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "detectgpt.jsonl"))
ap.add_argument("--window", type=int, default=256, help="words per scored window")
ap.add_argument("--k", type=int, default=3, help="windows per paper")
ap.add_argument("--n_perturb", type=int, default=20)
ap.add_argument("--span", type=int, default=2)
ap.add_argument("--pct", type=float, default=0.15)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
# The T5 mask filling samples, so the score moves with the seed; a run is reported as the
# mean over three seeds rather than as one draw.
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

dev = "cuda" if torch.cuda.is_available() else "cpu"
stok = AutoTokenizer.from_pretrained(a.scorer)
BF16_OK = (not torch.cuda.is_available()) or torch.cuda.get_device_capability(0)[0] >= 8
DT = torch.bfloat16 if BF16_OK else torch.float16
print('dtype:', DT, flush=True)
smod = AutoModelForCausalLM.from_pretrained(a.scorer, dtype=DT,
                                            device_map=dev, low_cpu_mem_usage=True).eval()
ptok = T5TokenizerFast.from_pretrained(a.perturber, model_max_length=512)
pmod = T5ForConditionalGeneration.from_pretrained(a.perturber, dtype=DT,
                                                  device_map=dev, low_cpu_mem_usage=True).eval()
print(f"scorer={a.scorer} perturber={a.perturber} on {dev}", flush=True)

PAT = re.compile(r"<extra_id_\d+>")

def mask_text(words, span, pct):
    n_spans = max(1, int(pct * len(words) / (span + 2)))
    out = list(words); placed = 0; guard = 0
    while placed < n_spans and guard < 200:
        guard += 1
        s = random.randint(1, max(1, len(out) - span - 1))
        if any(str(w).startswith("<extra_id_") for w in out[max(0, s-1):s+span+1]): continue
        out[s:s+span] = [f"<extra_id_{placed}>"]; placed += 1
    return " ".join(out), placed

@torch.no_grad()
def perturb_batch(texts, n_fill):
    enc = ptok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    stop = ptok.encode(f"<extra_id_{max(n_fill)-1}>")[0] if n_fill else 1
    out = pmod.generate(**enc, max_length=200, do_sample=True, top_p=0.96,
                        num_return_sequences=1, eos_token_id=stop)
    return ptok.batch_decode(out, skip_special_tokens=False)

def fill(masked, filled):
    parts = [p.strip() for p in PAT.split(filled)][1:]
    toks = masked.split(" "); res = []; i = 0
    for t in toks:
        if t.startswith("<extra_id_"):
            res.append(parts[i] if i < len(parts) else ""); i += 1
        else:
            res.append(t)
    return " ".join(w for w in res if w)

@torch.no_grad()
def logp(text):
    ids = stok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
    if ids.shape[1] < 16: return None
    lg = smod(ids).logits[:, :-1].float()
    return -F.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="mean").item()

done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
rows = [json.loads(l) for l in open(a.texts)]
if a.limit: rows = rows[:a.limit]
if a.nshard > 1: rows = [r for j, r in enumerate(rows) if j % a.nshard == a.shard]
todo = [r for r in rows if r["id"] not in done]
print(f"{len(done)} done, {len(todo)} to go", flush=True)

t0 = time.time()
with open(a.out, "a") as w:
    for i, r in enumerate(todo):
        words = r["text"].split()
        if len(words) < a.window: continue
        n = max(1, min(a.k, len(words) // a.window))
        starts = [int(j * (len(words) - a.window) / max(1, n - 1)) if n > 1 else 0 for j in range(n)]
        ds = []
        for s in starts:
            chunk = words[s:s + a.window]
            base = logp(" ".join(chunk))
            if base is None: continue
            masked, nf = [], []
            for _ in range(a.n_perturb):
                m, k = mask_text(chunk, a.span, a.pct); masked.append(m); nf.append(k)
            per_lp = []
            for b in range(0, len(masked), 10):
                try: outs = perturb_batch(masked[b:b+10], nf[b:b+10])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); continue
                for mk, fo in zip(masked[b:b+10], outs):
                    v = logp(fill(mk, fo))
                    if v is not None: per_lp.append(v)
            if len(per_lp) < 5: continue
            mu, sd = float(np.mean(per_lp)), float(np.std(per_lp))
            ds.append((base - mu) / (sd if sd > 1e-6 else 1e-6))
        if not ds: continue
        w.write(json.dumps({"id": r["id"], "year": r["year"], "source": r["source"],
                            "detectgpt": float(np.mean(ds)), "n_windows": len(ds),
                            "n_perturb": a.n_perturb}) + "\n")
        w.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(todo)}  {el/(i+1):.1f}s/paper  eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min", flush=True)
print("done", time.time() - t0, flush=True)
