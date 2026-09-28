"""NTS (Ma et al., ACL 2026): zero-shot detection by Normalized Temperature Sensitivity.

Faithful to the official repository (Shixuan-Ma/NTS, TeSeN/detector.py): surrogate
falcon-7b in bfloat16, input truncated at max_length=512, score = TS_norm =
Temperature_Sensitivity_sample(logits, labels, mask, T1=0.7, T2=1.4) exactly as in
compute_crit(mode='TS_norm'). Higher = more machine (paper: LLM text has higher TS).
The math below is copied verbatim; only the model path (HF hub id instead of a local
folder) and the jsonl I/O differ.
"""
import os, json, argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tiiuae/falcon-7b")
ap.add_argument("--texts", default=os.path.join(HERE, "texts.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "nts.jsonl"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(a.model,  # native Falcon class; the repo custom code predates transformers 5
                                             torch_dtype=torch.bfloat16).to(dev).eval()
tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.eos_token
print(f"M1={a.model} bf16 on {dev}", flush=True)


@torch.no_grad()
def logits_labels_mask(text):
    tokenized = tok(text, return_tensors="pt", return_token_type_ids=False,
                    return_attention_mask=True, truncation=True, max_length=512).to(dev)
    tokenized["input_ids"] = tokenized["input_ids"].long()
    labels = tokenized["input_ids"][:, 1:]
    attention_mask = tokenized["attention_mask"].to(torch.float16)[:, 1:]
    logits = model(**{k: v for k, v in tokenized.items() if k != "labels"}).logits[:, :-1]
    return logits, labels, attention_mask


def Temperature_Sensitivity_sample(logits_M1, labels, attention_mask, T1=0.6, T2=1.4):
    # verbatim from TeSeN/detector.py
    logits = logits_M1
    lprobs1 = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(lprobs1)
    logits = logits_M1 / T1
    lprobs1 = torch.log_softmax(logits, dim=-1)
    label_logprobs1 = lprobs1.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)
    logits = logits_M1 / T2
    lprobs2 = torch.log_softmax(logits, dim=-1)
    label_logprobs2 = lprobs2.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)
    per_token_ts = (lprobs1 - lprobs2)
    if (attention_mask.sum(dim=-1).item()) != 0:
        low_T_log_PPL = ((((((label_logprobs1)) * attention_mask))).sum(
            dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        high_T_log_PPL = ((((((label_logprobs2)) * attention_mask))).sum(
            dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        TS = abs(low_T_log_PPL - high_T_log_PPL)
        E = (((per_token_ts * probs).sum(dim=-1)) * attention_mask).sum(
            dim=-1).sum(dim=-1).item() / (attention_mask.sum(dim=-1).item())
        per_token_std = (((per_token_ts) ** 2) * probs).sum(dim=-1) - ((per_token_ts * probs).sum(dim=-1)) ** 2
        std = (((((per_token_std.sqrt()) * attention_mask))).sum(
            dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        Ts = (TS - E) / std
    else:
        Ts = -10
    return Ts


done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
rows = [json.loads(l) for l in open(a.texts)]
if a.limit: rows = rows[:a.limit]
todo = [r for r in rows if r["id"] not in done]
print(f"{len(done)} done, {len(todo)} to go", flush=True)

t0 = time.time()
with open(a.out, "a") as w:
    for i, r in enumerate(todo):
        logits, labels, mask = logits_labels_mask(r["text"])
        if labels.shape[1] < 16:
            continue
        # compute_crit(mode='TS_norm') calls with T1=0.7, T2=1.4
        ts = Temperature_Sensitivity_sample(logits, labels, mask, T1=0.7, T2=1.4)
        w.write(json.dumps({"id": r["id"], "year": r.get("year", 0), "source": r.get("source", ""),
                            "nts": ts, "n_tokens": int(labels.shape[1]) + 1}) + "\n")
        w.flush()
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(todo)}  {(time.time()-t0)/(i+1):.2f}s/paper", flush=True)
print("done", time.time() - t0, flush=True)
