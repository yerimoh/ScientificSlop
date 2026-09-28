"""Log-probability scoring with a local causal LM, for position-free pairwise measures.

Used by Argument_Graph. This is not an LLM judgment: nothing is asked of the model, no answer is
parsed. The only thing read off is the probability the model assigns to one sentence given another,
with the two sentences concatenated by the caller in a fixed order. Whatever order the sentences had
in the source document is therefore not an input.

    pmi(Y -> X) = mean log P(X | Y) - mean log P(X)      nats per token of X

The first token of X is skipped in both terms, so the conditional and the unconditional side score
the same tokens. Results are cached on disk per (model, list of texts), so a rerun over the same
introduction costs nothing and a change of scorer invalidates the cache by construction.

Model: `PMI_MODEL` env, default Qwen/Qwen2.5-7B-Instruct, bf16 on CUDA, fp32 on CPU. Weights are
read from the local HuggingFace cache; nothing is downloaded at measurement time (HF_HUB_OFFLINE).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import time

MODEL = os.environ.get("PMI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pmi_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_tok = _mdl = _dev = None


def available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _tok, _mdl, _dev
    if _mdl is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16 if _dev == "cuda" else torch.float32).to(_dev).eval()
    return _tok, _mdl, _dev


def device() -> str:
    _load()
    return _dev


def logp_batch(prefixes, targets, bs: int = 16):
    """Mean log P(target tokens | prefix) per pair, the target's first token skipped."""
    import torch
    tok, mdl, dev = _load()
    out = []
    with torch.no_grad():
        for k in range(0, len(targets), bs):
            rows = []
            for p, t in zip(prefixes[k:k + bs], targets[k:k + bs]):
                pi = tok(p, add_special_tokens=False)["input_ids"] if p else []
                ti = tok(t, add_special_tokens=False)["input_ids"]
                rows.append((pi, ti))
            L = max(len(a) + len(b) for a, b in rows)
            ids = torch.full((len(rows), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            spans = []
            for r, (a, b) in enumerate(rows):
                s = a + b
                ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
                spans.append((len(a), len(s)))
            lg = torch.log_softmax(mdl(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(), -1)
            ids = ids.to(dev)
            for r, (a, b) in enumerate(spans):
                st = a + 1
                if b - st < 1:
                    out.append(float("nan")); continue
                tgt = ids[r, st:b]
                lp = lg[r, st - 1:b - 1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                out.append(lp.mean().item())
    return out


# --------------------------------------------------------------------------- warm worker
# `pmi_daemon.py` keeps the model loaded on a GPU. The exchange is the cache directory itself: the
# request carries the texts, the daemon computes the same matrix through the same function, and the
# file it writes is the file this process was already going to look for.
_REQ = os.path.join(CACHE_DIR, "_requests")


def _daemon_ready() -> bool:
    if os.environ.get("AG_PMI_DAEMON", "1") != "1":
        return False
    return os.path.exists(os.path.join(_REQ, "READY"))


def _via_daemon(key: str, texts, cp: str, wait: float = None):
    wait = wait or float(os.environ.get("AG_PMI_WAIT", "900"))
    os.makedirs(_REQ, exist_ok=True)
    rp = os.path.join(_REQ, key + ".json")
    if not os.path.exists(cp):
        tmp = rp + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"texts": list(texts)}, f)
        os.replace(tmp, rp)
    t0 = time.time()
    while time.time() - t0 < wait:
        if os.path.exists(cp):
            try:
                M = json.load(open(cp))
            except Exception:
                time.sleep(0.5); continue
            return [[float("nan") if v is None else v for v in row] for row in M]
        if not _daemon_ready():
            break
        time.sleep(1.0)
    return None


def pmi_matrix(texts, bs: int = 16):
    """M[j][i] = pmi(texts[j] -> texts[i]); diagonal is NaN. Cached per (model, texts)."""
    key = hashlib.sha256((MODEL + "\n" + "\n\x1e".join(texts)).encode()).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        M = json.load(open(cp))
        return [[float("nan") if v is None else v for v in row] for row in M]
    if _daemon_ready():                       # a warm worker holds the model; ask it and read the cache it writes
        M = _via_daemon(key, texts, cp)
        if M is not None:
            return M
    n = len(texts)
    base = logp_batch([""] * n, texts, bs=bs)
    pre, tgt, idx = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pre.append(texts[j].rstrip() + " "); tgt.append(texts[i]); idx.append((j, i))
    joint = logp_batch(pre, tgt, bs=bs)
    M = [[float("nan")] * n for _ in range(n)]
    for (j, i), v in zip(idx, joint):
        M[j][i] = round(v - base[i], 5)       # rounded here too, so a fresh call and a cache hit return the same numbers
    json.dump([[None if (isinstance(v, float) and math.isnan(v)) else v for v in row] for row in M], open(cp, "w"))
    return M
