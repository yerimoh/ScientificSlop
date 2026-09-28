"""Binoculars (Hans et al., ICML 2024) on the ICLR 2017-2025 corpus.

score B(x) = log-PPL of x under the performer
             ---------------------------------
             cross-perplexity between observer and performer
Lower B = more machine-like. Paper pairs a base model with its instruct sibling
(falcon-7b / falcon-7b-instruct); we use the cached Qwen2.5-7B pair, which has the
same base/instruct relation. Per paper we score K windows of 512 tokens and average.
"""
import os, json, argparse, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--observer", default="Qwen/Qwen2.5-7B")
ap.add_argument("--performer", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--texts", default=os.path.join(HERE, "texts.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "binoculars.jsonl"))
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--k", type=int, default=6, help="windows per paper")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args()

ng = torch.cuda.device_count()
dev = "cuda:0" if ng else "cpu"
dev2 = "cuda:1" if ng > 1 else dev          # keep the pair off one card when we have two
tok = AutoTokenizer.from_pretrained(a.observer)
bf16_ok = (not torch.cuda.is_available()) or torch.cuda.get_device_capability(0)[0] >= 8
kw = dict(dtype=torch.bfloat16 if bf16_ok else torch.float16, low_cpu_mem_usage=True)
print("dtype:", "bf16" if bf16_ok else "fp16 (pre-Ampere GPU)", flush=True)
obs = AutoModelForCausalLM.from_pretrained(a.observer, **kw).to(dev).eval()
per = AutoModelForCausalLM.from_pretrained(a.performer, **kw).to(dev2).eval()
print(f"observer={a.observer} on {dev}   performer={a.performer} on {dev2}   ngpu={ng}", flush=True)

@torch.no_grad()
def binoculars(ids):
    """ids: LongTensor [1, T]. Returns (B, ppl, xppl)."""
    lo = obs(ids.to(dev)).logits[:, :-1].float()
    lp = per(ids.to(dev2)).logits[:, :-1].float().to(dev)
    tgt = ids[:, 1:].to(dev)
    # log-perplexity of the text under the performer
    ppl = F.cross_entropy(lp.transpose(1, 2), tgt, reduction="mean").item()
    # cross-perplexity: expected surprisal of the observer's distribution under the performer
    xppl = -(lo.softmax(-1) * lp.log_softmax(-1)).sum(-1).mean().item()
    return ppl / xppl, ppl, xppl

done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
rows = [json.loads(l) for l in open(a.texts)]
if a.limit: rows = rows[:a.limit]
if a.nshard > 1: rows = [r for j, r in enumerate(rows) if j % a.nshard == a.shard]
todo = [r for r in rows if r["id"] not in done]
print(f"{len(done)} already done, {len(todo)} to go", flush=True)

t0 = time.time()
with open(a.out, "a") as w:
    for i, r in enumerate(todo):
        enc = tok(r["text"], return_tensors="pt", truncation=False).input_ids[0]
        T = enc.shape[0]
        if T < 64:
            continue
        n = max(1, min(a.k, T // a.window))
        starts = [int(j * (T - a.window) / max(1, n - 1)) if n > 1 else 0 for j in range(n)]
        vals = []
        for s in starts:
            ids = enc[s:s + a.window].unsqueeze(0)
            if ids.shape[1] < 64: continue
            try: vals.append(binoculars(ids))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
        if not vals: continue
        B = sum(v[0] for v in vals) / len(vals)
        w.write(json.dumps({"id": r["id"], "year": r["year"], "source": r["source"],
                            "binoculars": B, "ppl": sum(v[1] for v in vals)/len(vals),
                            "xppl": sum(v[2] for v in vals)/len(vals),
                            "n_windows": len(vals), "n_tokens": int(T)}) + "\n")
        w.flush()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(todo)}  {el/(i+1):.2f}s/paper  eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min", flush=True)
print("done", time.time() - t0, flush=True)
