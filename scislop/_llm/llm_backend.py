"""Unified open-source LLM backend — ONE entry point for every mold's
MEASUREMENT / EXTRACTION / JUDGE call.

Design split (2026-07-14):
  * MEASUREMENT / EXTRACTION / JUDGE  -> open-source **Qwen2.5-32B-Instruct** (vLLM, GPU).
    Reproducible, free, batched, high concurrency. This module.
  * PAPER REWRITE / FILE EDITING      -> **Haiku CLI agent** (unchanged).
    The rewriting AI is the experimental subject of the survival loop, so it stays as-is.

Server:  bash _llm/serve_qwen.sh   (launched via `sr 4 48 --qos=${SLURM_QOS}`)
         writes its reachable URL into _llm/llm_endpoint.txt

Usage (drop-in for the old `claude(prompt)` helper):
    from llm_backend import llm
    out = llm("...prompt...")                    # -> str
    obj = llm_json("...return JSON...")          # -> dict/list (robust parse)
"""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ENDPOINT_FILE = os.path.join(HERE, "llm_endpoint.txt")
MODEL = os.environ.get("LLM_MODEL", "qwen")          # served-model-name in serve_qwen.sh
DEFAULT_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))


def endpoint() -> str:
    """Resolve the vLLM OpenAI-compatible base URL (env wins, else the file the server wrote)."""
    ep = os.environ.get("LLM_ENDPOINT")
    if ep:
        return ep.rstrip("/")
    if os.path.exists(ENDPOINT_FILE):
        return open(ENDPOINT_FILE).read().strip().rstrip("/")
    raise RuntimeError(
        "No LLM endpoint. Start the server:\n"
        "  screen -dmS qwen_server bash -c 'sr 4 48 --qos=${SLURM_QOS} bash _llm/serve_qwen.sh'\n"
        "or export LLM_ENDPOINT=http://<host>:8765/v1")


def is_up(timeout: int = 5) -> bool:
    try:
        urllib.request.urlopen(f"{endpoint()}/models", timeout=timeout).read()
        return True
    except Exception:
        return False


def wait_until_up(max_wait: int = 1800, poll: int = 10) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if is_up():
            return True
        time.sleep(poll)
    return False


def llm(prompt: str, system: str | None = None, model: str = MODEL,
        max_tokens: int = 2048, temperature: float = 0.0,
        timeout: int = DEFAULT_TIMEOUT, retries: int = 4) -> str:
    """Chat completion. Greedy by default (temperature=0) for reproducible measurement."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": msgs,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    backoff = [3, 8, 20, 40]
    last = ""
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{endpoint()}/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            out = r["choices"][0]["message"]["content"] or ""
            if out.strip():
                return out
            last = out
        except Exception as e:  # server still loading / transient
            last = f"__ERR__{e}"
        if i < retries - 1:
            time.sleep(backoff[min(i, len(backoff) - 1)])
    return last if not last.startswith("__ERR__") else ""


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def llm_json(prompt: str, system: str | None = None, default=None, **kw):
    """Chat completion that must return JSON. Strips ``` fences, grabs the outermost object."""
    raw = llm(prompt, system=system, **kw)
    if not raw:
        return default
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = _JSON_RE.search(txt)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return default


# ---- Back-compat shim -------------------------------------------------------
# Old mold code did `from temp_common import claude` (Haiku CLI, text in/out).
# Those call sites are MEASUREMENT calls, so they now route here transparently.
def claude(prompt: str, model: str = None, timeout: int = DEFAULT_TIMEOUT,
           retries: int = 4, **_) -> str:
    """Deprecated name kept so existing measurement call sites keep working."""
    return llm(prompt, timeout=timeout, retries=retries)


if __name__ == "__main__":
    import sys
    print("endpoint:", endpoint(), "| up:", is_up())
    if is_up():
        print(llm(sys.argv[1] if len(sys.argv) > 1 else "Reply with exactly: OK"))
