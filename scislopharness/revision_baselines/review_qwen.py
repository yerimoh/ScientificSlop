"""Generic reviewer for arm a3_review: B3's REVIEW_PROMPT on Qwen2.5-32B (vLLM), same
truncation ladder and schema check as b3_run.py. R1 reuses B3's stored review of the same
R0 manuscript when present."""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import B3_RUNS, LLM_DIR, paper_text  # noqa: E402
from prompts import REVIEW_PROMPT  # noqa: E402
sys.path.insert(0, str(LLM_DIR))

TRUNC = 50000


def valid_review(o):
    if not isinstance(o, dict):
        return False
    if not (isinstance(o.get('summary'), str) and o['summary'].strip()):
        return False
    for k in ('strengths', 'weaknesses', 'questions'):
        v = o.get(k)
        if not (isinstance(v, list) and len(v) >= 1 and all(isinstance(x, str) and x.strip() for x in v)):
            return False
    r = o.get('rating')
    return isinstance(r, (int, float)) and 1 <= r <= 10


def _extract_json(txt):
    if not txt:
        return None
    txt = txt.replace('```json', '').replace('```', '')
    start = txt.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(txt)):
        if txt[i] == '{':
            depth += 1
        elif txt[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:i + 1])
                except Exception:
                    return None
    return None


def qwen_up() -> bool:
    try:
        import llm_backend
        return llm_backend.is_up()
    except Exception:
        return False


def b3_r1_review(code: str):
    p = B3_RUNS / code / 'logs' / f'{code}_b3_R1_review.json'
    if p.exists():
        try:
            o = json.load(open(p))
            if valid_review(o):
                return o
        except Exception:
            pass
    return None


def get_review(code: str, n: int, tree: Path, logdir: Path):
    """Returns (review, source) or (None, reason)."""
    logdir.mkdir(parents=True, exist_ok=True)
    cache = logdir / f'{code}_R{n}_review.json'
    if cache.exists():
        try:
            o = json.load(open(cache))
            if valid_review(o.get('review')):
                return o['review'], o['source']
        except Exception:
            pass
    if n == 1:
        o = b3_r1_review(code)
        if o:
            json.dump({'review': o, 'source': 'b3_reuse'}, open(cache, 'w'), indent=1)
            return o, 'b3_reuse'
    if not qwen_up():
        return None, 'qwen_down'
    import llm_backend
    tex = paper_text(tree)
    for lim in (TRUNC, 24000, 14000):
        try:
            obj = _extract_json(llm_backend.llm(REVIEW_PROMPT + tex[:lim], timeout=600))
        except Exception:
            obj = None
        if obj is not None and valid_review(obj):
            json.dump({'review': obj, 'source': f'qwen@{lim}'}, open(cache, 'w'), indent=1)
            return obj, f'qwen@{lim}'
    return None, 'qwen_invalid'
