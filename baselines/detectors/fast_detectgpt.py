"""Fast-DetectGPT (Bao et al., ICLR 2024), faithful implementation.

Criterion = analytic sampling discrepancy, copied verbatim from the official
scripts/fast_detect_gpt.py (get_sampling_discrepancy_analytic). Higher = more machine.
Paper black-box setting: sampling gpt-j-6B, scoring gpt-neo-2.7B, both float16 per the
official model.py float16_models list. The repo's later recommendation falcon-7b /
falcon-7b-instruct is run as a second configuration.

Only forced adaptation for full papers: the original feeds whole passages with no
truncation (their data are short); our documents exceed the 2048-token context of
these models, so the tokenizer truncates at model max context (first 2048 tokens).
Nothing else is changed.
"""
import os, json, argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--sampling", default="EleutherAI/gpt-j-6b")
ap.add_argument("--scoring", default="EleutherAI/gpt-neo-2.7B")
ap.add_argument("--texts", default=os.path.join(HERE, "texts.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "fast_detectgpt.jsonl"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

ng = torch.cuda.device_count()
dev = "cuda:0" if ng else "cpu"
dev2 = "cuda:1" if ng > 1 else dev

# official model.py: these models load in float16
kw = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
stok = AutoTokenizer.from_pretrained(a.scoring)
score_m = AutoModelForCausalLM.from_pretrained(a.scoring, **kw).to(dev).eval()
if a.sampling != a.scoring:
    rtok = AutoTokenizer.from_pretrained(a.sampling)
    ref_m = AutoModelForCausalLM.from_pretrained(a.sampling, **kw).to(dev2).eval()
else:
    rtok, ref_m = stok, score_m
MAXLEN = min(getattr(score_m.config, "max_position_embeddings", 2048) or 2048, 2048)
print(f"sampling={a.sampling} on {dev2}  scoring={a.scoring} on {dev}  maxlen={MAXLEN}", flush=True)


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    # verbatim from the official repository
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


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
        tokenized = stok(r["text"], return_tensors="pt", return_token_type_ids=False,
                         truncation=True, max_length=MAXLEN).to(dev)
        if tokenized.input_ids.shape[1] < 16:
            continue
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = score_m(**tokenized).logits[:, :-1].float()
            if a.sampling == a.scoring:
                logits_ref = logits_score
            else:
                tok2 = rtok(r["text"], return_tensors="pt", return_token_type_ids=False,
                            truncation=True, max_length=MAXLEN).to(dev2)
                assert torch.all(tok2.input_ids[:, 1:].to(dev) == labels), "Tokenizer is mismatch."
                logits_ref = ref_m(**tok2).logits[:, :-1].float().to(dev)
            crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        w.write(json.dumps({"id": r["id"], "year": r.get("year", 0), "source": r.get("source", ""),
                            "fast_detectgpt": crit, "n_tokens": int(tokenized.input_ids.shape[1])}) + "\n")
        w.flush()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(todo)}  {el/(i+1):.2f}s/paper", flush=True)
print("done", time.time() - t0, flush=True)
