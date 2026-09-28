"""LLM access for the checkers that need extraction (never for judging quality).

Routes through artifact-ai2science/_llm/llm_backend.py (Qwen2.5-32B-Instruct on vLLM, greedy).
Rules carried over from the mold program:
  * measurement never calls a paid closed model; the Haiku CLI is reserved for the rewriting arm;
  * greedy decoding; where a scalar is read off the output we run n_runs (default 3) and keep the
    run whose scalar is the median (server-side batching makes greedy runs not byte-identical);
  * every quote the model returns is verified verbatim against the source; unverifiable quotes
    are dropped and counted (hallucination rate), because LLM judges were shown to favour AI
    polish and to invent evidence;
  * all calls are cached on disk keyed by sha256(prompt) so reruns are free and reproducible.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
try:
    import llm_backend as _B
except Exception:  # pragma: no cover
    _B = None

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def is_up() -> bool:
    try:
        return bool(_B) and _B.is_up()
    except Exception:
        return False


MODEL_TAG = os.environ.get("MODEL_TAG", "qwen32b")   # cache namespace: never reuse one model's answers for another


def _key(prompt: str, system: str | None, tag: str) -> str:
    base = tag + "\n" + (system or "") + "\n" + prompt
    if MODEL_TAG == "qwen32b":
        # legacy key format: everything cached before 0912 was produced by Qwen2.5-32B
        return hashlib.sha256(base.encode()).hexdigest()
    return hashlib.sha256((MODEL_TAG + "|" + base).encode()).hexdigest()


def llm_json(prompt: str, system: str | None = None, tag: str = "", max_tokens: int = 4096, run: int = 0):
    """One cached JSON call. `run` distinguishes repeated runs of the same prompt."""
    k = _key(prompt, system, f"{tag}#{run}")
    cp = os.path.join(CACHE_DIR, k + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    if not is_up():
        raise RuntimeError("LLM endpoint is down; start _llm/serve_qwen.sh (see RUN.md) or use --dry-run")
    out = _B.llm_json(prompt, system=system, default=None, max_tokens=max_tokens)
    json.dump(out, open(cp, "w"), ensure_ascii=False)
    return out


def llm_json_median(prompt: str, scalar, system: str | None = None, tag: str = "", n_runs: int = 3, **kw):
    """Run n_runs times, pick the output whose scalar(output) is the median. Returns (obj, all_scalars)."""
    outs, vals = [], []
    for r in range(n_runs):
        o = llm_json(prompt, system=system, tag=tag, run=r, **kw)
        outs.append(o)
        try:
            vals.append(scalar(o))
        except Exception:
            vals.append(None)
    good = [(v, o) for v, o in zip(vals, outs) if v is not None]
    if not good:
        return None, vals
    med = statistics.median([v for v, _ in good])
    best = min(good, key=lambda vo: abs(vo[0] - med))
    return best[1], vals


# --------------------------------------------------------------------------- verbatim checks

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\\[a-z]+\*?", " ", s)
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_quote(quote: str, source: str, min_chars: int = 25) -> tuple[int, int] | None:
    """Locate `quote` in `source` after light normalisation. Returns a [start, end) span in the
    source or None. Long quotes may be matched by their first 2*min_chars normalised characters."""
    if not quote or not source:
        return None
    i = source.find(quote)
    if i >= 0:
        return (i, i + len(quote))
    nq, ns = _norm(quote), _norm(source)
    if not nq:
        return None
    j = ns.find(nq)
    if j < 0 and len(nq) > 2 * min_chars:
        j = ns.find(nq[:min_chars * 2])
    if j < 0:
        return None
    words_before = ns[:j].count(" ") if j > 0 else 0
    tok_spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9%]+", source)]
    if words_before >= len(tok_spans):
        return None
    start = tok_spans[words_before][0]
    n_words = max(1, nq.count(" ") + 1)
    endi = min(len(tok_spans) - 1, words_before + n_words - 1)
    return (start, tok_spans[endi][1])


def verify_quotes(obj, source: str, fields=("quote", "evidence", "claim_quote", "reason_quote", "description_quote")) -> dict:
    """Walk a JSON object; for every string under one of `fields`, attach `<field>_span` or mark
    it unverifiable. Returns {"n": ..., "unverified": ...}."""
    stats = {"n": 0, "unverified": 0}

    def walk(x):
        if isinstance(x, dict):
            for k in list(x.keys()):
                v = x[k]
                if k in fields and isinstance(v, str):
                    stats["n"] += 1
                    sp = find_quote(v, source)
                    x[k + "_span"] = list(sp) if sp else None
                    if not sp:
                        stats["unverified"] += 1
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return stats
