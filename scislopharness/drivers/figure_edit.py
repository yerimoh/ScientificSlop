"""Erases only the expository elements from a method diagram. It does not redraw.

Why this approach. On 0919 the editor was asked to redraw the diagram in TikZ and the gate reverted four times in a row.
The reason differed each time but the root was one. A redraw has to reproduce the whole topology of boxes and edges, which
is unrelated to the defect being fixed and damages the mechanism when it fails. Only the expository phrases inside the diagram
need fixing, so the right move is to keep the original pixels and erase just where those phrases are.

Procedure
  1. The measurer already decides which phrases to erase (the lines flagged by fig_exposition).
  2. Where each phrase sits in the figure is asked, as coordinates, of the same VLM used for transcription.
  3. That rectangle is filled with the surrounding background colour. No pixel is newly drawn. No generative model is used,
     so there is no room for fabrication.
  4. The erased figure is re-transcribed with the same prompt and rescored with the same verdict function.

  python3 figure_edit.py <code> <in.jpg> <out.jpg> --drop "phrase1" --drop "phrase2"
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VLM_SCRIPTS = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/benchA4S/scripts")

LOCATE_PROMPT = (
    'Locate each phrase below in this image and output its bounding box.\n\n'
    'Phrases:\n{phrases}\n\n'
    'Output JSON only: [{{"bbox_2d": [x1, y1, x2, y2], "label": "the phrase"}}]. The box must cover the whole '
    'phrase, all of its characters top to bottom including descenders, and nothing else. Omit a phrase you cannot '
    'find.')

ZOOM_PROMPT = (
    'Locate the phrase "{phrase}" in this image crop and output its bounding box.\n\n'
    'Output JSON only: [{{"bbox_2d": [x1, y1, x2, y2], "label": "the phrase"}}]. The box must cover the whole '
    'phrase, all of its characters top to bottom, and nothing else. Output an empty list if the phrase is not here.')

MAX_NEW = 1500


def _ask(img, prompt: str) -> str:
    """Same call as the transcription pipeline. Qwen2.5-VL-32B, bf16, greedy."""
    import torch
    sys.path.insert(0, str(VLM_SCRIPTS))
    import vision_calls as V
    m, proc = V.model()
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors='pt').to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]


def _json(s: str) -> dict:
    s = re.sub(r'^```(?:json)?|```$', '', (s or '').strip(), flags=re.M)
    m = re.search(r'\{.*\}', s, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _load(path: Path):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if max(im.size) > 2000:
        sc = 2000.0 / max(im.size)
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    return im


def _boxes_from(out: str) -> list[dict]:
    """Qwen2.5-VL grounding output. Arrives as a list or an object; the key is bbox_2d."""
    t = re.sub(r'^```(?:json)?|```$', '', (out or '').strip(), flags=re.M)
    for pat in (r'\[.*\]', r'\{.*\}'):
        m = re.search(pat, t, re.S)
        if not m:
            continue
        try:
            j = json.loads(m.group(0))
        except Exception:
            continue
        items = j if isinstance(j, list) else (j.get('boxes') or j.get('objects') or [j])
        out2 = []
        for it in items:
            if not isinstance(it, dict):
                continue
            bb = it.get('bbox_2d') or it.get('box_2d') or it.get('bbox')
            if bb and len(bb) == 4:
                out2.append({'phrase': it.get('label') or it.get('phrase'), 'box_2d': [float(v) for v in bb]})
        if out2:
            return out2
    return []


def locate(image_path: Path, phrases: list[str], workdir: Path | None = None) -> list[dict]:
    """Locates phrases with the settled method. Instead of asking the model for coordinates, it picks among line candidates cut by pixels.

    On 0919 asking for coordinates directly was tried three times and missed three times, once clipping a box of the mechanism. Details in
    ../figure_edit_study/README.md."""
    import figure_bands as B
    wd = Path(workdir) if workdir else Path(image_path).parent / '_bands'
    return B.locate_phrases(Path(image_path), phrases, wd)


def _refine(a, box, grow=0.9, min_pad=8):
    """A box from the VLM often covers only part of the text. Grow the box generously, then find how far the pixels that differ
    from the background (the ink of the text) actually extend inside it, and reset the box to that extent. The decision uses
    pixel values only, so it is deterministic, and the growth stops only within the text blob."""
    import numpy as np
    H, W = a.shape[:2]
    x1, y1, x2, y2 = box
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    gx, gy = int(max(min_pad, w * 0.10)), int(max(min_pad, h * grow))
    X1, Y1 = max(0, x1 - gx), max(0, y1 - gy)
    X2, Y2 = min(W, x2 + gx), min(H, y2 + gy)
    reg = a[Y1:Y2, X1:X2].astype(int)
    if reg.size == 0:
        return [x1, y1, x2, y2]
    flat = reg.reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    ink = (np.abs(reg - bg).sum(2) > 60)
    if not ink.any():
        return [x1, y1, x2, y2]
    rows = np.where(ink.any(1))[0]
    cols = np.where(ink.any(0))[0]
    # Follow only the ink blob overlapping the original box. Stop where two or more empty rows appear above or below.
    r0 = y1 - Y1
    up = r0
    while up > 0 and ink[max(0, up - 1)].any():
        up -= 1
    dn = min(len(ink) - 1, (y2 - Y1))
    while dn < len(ink) - 1 and ink[min(len(ink) - 1, dn + 1)].any():
        dn += 1
    lo = max(0, min(cols.min(), x1 - X1))
    hi = min(reg.shape[1], max(cols.max() + 1, x2 - X1))
    return [int(X1 + lo), int(Y1 + up), int(X1 + hi), int(Y1 + dn + 1)]


def erase(image_path: Path, boxes: list[dict], out_path: Path, pad: int = 3) -> dict:
    """Fills the rectangles with the surrounding background colour. No generation, only overwriting."""
    from PIL import Image
    import numpy as np
    im = Image.open(image_path).convert('RGB')
    a = np.array(im)
    H, W = a.shape[:2]
    done = []
    for b in boxes:
        raw = [int(round(v)) for v in b['box_2d']]
        raw = [max(0, min(raw[0], W - 1)), max(0, min(raw[1], H - 1)),
               max(1, min(raw[2], W)), max(1, min(raw[3], H))]
        x1, y1, x2, y2 = _refine(a, raw)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        ring = []
        for yy in (max(0, y1 - 2), min(H - 1, y2 + 1)):
            ring.append(a[yy, x1:x2])
        for xx in (max(0, x1 - 2), min(W - 1, x2 + 1)):
            ring.append(a[y1:y2, xx])
        ring = np.concatenate([r.reshape(-1, 3) for r in ring if r.size]) if ring else None
        if ring is None or ring.size == 0:
            fill = np.array([255, 255, 255], dtype=np.uint8)
        else:
            vals, counts = np.unique(ring, axis=0, return_counts=True)
            fill = vals[counts.argmax()]
        a[y1:y2, x1:x2] = fill
        done.append({'phrase': b.get('phrase'), 'box_vlm': [int(v) for v in raw],
                     'box_used': [int(x1), int(y1), int(x2), int(y2)], 'fill': [int(v) for v in fill]})
    Image.fromarray(a).save(out_path, quality=95)
    return {'n_erased': len(done), 'erased': done, 'size': [int(W), int(H)]}


TR_CACHE = Path(__file__).resolve().parent.parent / 'transcript_cache'


def transcribe(image_path: Path) -> list[str]:
    """Cached by image hash. Loading the 32B model from disk is almost the whole cost of this call."""
    import hashlib
    TR_CACHE.mkdir(parents=True, exist_ok=True)
    ck = TR_CACHE / (hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:16] + '.json')
    if ck.exists():
        try:
            return json.loads(ck.read_text())
        except Exception:
            pass
    out = _transcribe_uncached(image_path)
    try:
        ck.write_text(json.dumps(out, ensure_ascii=False))
    except Exception:
        pass
    return out


def _transcribe_uncached(image_path: Path) -> list[str]:
    """Re-transcribes with the same prompt and model. Values are comparable only with the same tool the measurer reads with."""
    P = ('Transcribe every piece of text that appears in this figure, verbatim, including numbers, symbols, '
         'labels inside boxes, axis labels, legend entries and small annotations. Do not describe the figure and do '
         'not add anything that is not written in it. Return JSON: {"lines": ["...", "..."]}')
    out = _json(_ask(_load(image_path), P))
    return (out or {}).get('lines', [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--drop', action='append', default=[])
    ap.add_argument('--report', default='')
    ap.add_argument('--method', default=os.environ.get('SH_FIG_METHOD', 'lama'),
                    choices=['lama', 'mask'])
    ap.add_argument('--prior-lines', default='',
                    help='JSON list. Transcription of the pre-edit figure. When re-editing an already edited figure, pass the '
                         'lines_after of the previous EDIT_REPORT. Read from the cache if absent')
    a = ap.parse_args()
    wd = Path(a.dst).parent / '_bands'
    boxes = locate(Path(a.src), a.drop, wd)
    boxes = [shrink_stage_prefix(b) for b in boxes]
    rec = {'requested': a.drop, 'located': [b.get('phrase') for b in boxes], 'method': a.method}
    if not boxes:
        rec.update({'status': 'not_located', 'n_erased': 0})
    elif a.method == 'lama':
        # Mask inpainting. It continues the surrounding texture, so unlike a fill it leaves no rectangular mark.
        import figure_inpaint as FI
        rec.update(FI.inpaint(Path(a.src), [b['box_2d'] for b in boxes], Path(a.dst)))
        rec['n_erased'] = len(boxes)
        rec['erased'] = [{'phrase': b.get('phrase'), 'box_used': [int(v) for v in b['box_2d']]} for b in boxes]
    else:
        rec.update(erase(Path(a.src), boxes, Path(a.dst)))
    if Path(a.dst).exists():
        rec['preservation'] = _preservation(Path(a.src), Path(a.dst),
                                            [[int(v) for v in b['box_2d']] for b in boxes])
        if os.environ.get('SH_FIG_RETRANSCRIBE', '0') == '1':
            rec['lines_after'] = transcribe(Path(a.dst))
            rec['lines_source'] = 'retranscribed'
        else:
            # Derive the transcription by removing the erased lines from the list. Those pixels were filled with background and
            # the outside-target change ratio proves it, so there is no reason to reload the 32B model to confirm the same conclusion.
            import figure_bands as _B
            base = json.load(open(a.prior_lines)) if a.prior_lines and Path(a.prior_lines).exists() else _prior_lines(Path(a.src), a.code)
            gone = {_B._norm(b.get('phrase') or '') for b in boxes}
            keep_rest = {_B._norm(b.get('phrase') or ''): STAGE_RE.match(str(b.get('phrase') or '')).group(2).strip()
                         for b in boxes if b.get('prefix_only')}
            rec['lines_after'] = [(keep_rest[_B._norm(l)] if _B._norm(l) in keep_rest else l) for l in base
                                  if _B._norm(l) not in gone or _B._norm(l) in keep_rest]
            rec['lines_source'] = 'derived by removing the erased lines'
    if a.report:
        Path(a.report).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k != 'lines_after'}, ensure_ascii=False))


STAGE_RE = re.compile(r'^\s*((?:stage|step|phase)\s*[0-9ivx]+\s*[:.)]?)\s*(.+)$', re.I)


def shrink_stage_prefix(b: dict) -> dict:
    """In a stage heading like 'Step 2: Prefix-Primed Continuation' only the numbered prefix ("Step 2:") is expository; the rest names a component.
    On 0920 FA0050 the whole line was erased and the box lost its name. Cut the rectangle from the left by the prefix's share of characters only.
    Assumes a single line in one font, with a little extra margin."""
    p = str(b.get('phrase') or '')
    m = STAGE_RE.match(p)
    if not m or not m.group(2).strip():
        return b
    frac = min(0.9, (len(m.group(1)) + 0.5) / max(1, len(p)))
    x1, y1, x2, y2 = b['box_2d']
    nb = dict(b); nb['box_2d'] = [x1, y1, x1 + (x2 - x1) * frac, y2]; nb['prefix_only'] = m.group(1)
    return nb


def _prior_lines(src: Path, code: str = '') -> list:
    """Transcription of the pre-edit figure. Order: (1) this pipeline's transcription cache, (2) the measurer's cache of the original
    diagram transcription (present for every bench paper), (3) one transcription with the Claude reader model. Qwen 32B is not loaded here.
    Early on 0920 FA0004 skipped (2) and stalled for an hour trying to load Qwen on the CPU."""
    import hashlib
    ck = TR_CACHE / (hashlib.md5(Path(src).read_bytes()).hexdigest()[:16] + '.json')
    if ck.exists():
        try:
            return json.loads(ck.read_text())
        except Exception:
            pass
    if code:
        try:
            import extra_items as X
            lines = X._transcript(code)
            if lines:
                return lines
        except Exception:
            pass
    if os.environ.get('SH_FIG_READER', 'claude') == 'claude':
        import figure_bands as B
        out = B._ask(_load(src), 'Transcribe every piece of text that appears in this figure, verbatim, including numbers, '
                     'symbols, labels inside boxes, axis labels, legend entries and small annotations. Do not describe the '
                     'figure. Return JSON: {"lines": ["...", "..."]}', Path(src).parent / '_bands')
        lines = (_json(out) or {}).get('lines', [])
        try:
            TR_CACHE.mkdir(parents=True, exist_ok=True); ck.write_text(json.dumps(lines, ensure_ascii=False))
        except Exception:
            pass
        return lines
    return transcribe(src)


def _preservation(orig: Path, edited: Path, targets, thr: int = 24) -> dict:
    """How much changed outside the targets. A step that touches the figure is accepted only when this value is small."""
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(orig).convert('RGB')).astype(int)
    b = np.array(Image.open(edited).convert('RGB')).astype(int)
    if a.shape != b.shape:
        return {'comparable': False}
    d = (np.abs(a - b).sum(2) > thr)
    m = np.zeros(d.shape, bool)
    for x1, y1, x2, y2 in targets:
        m[max(0, y1):y2, max(0, x1):x2] = True
    return {'comparable': True, 'changed_inside': int((d & m).sum()),
            'changed_outside': int((d & ~m).sum()),
            'outside_ratio': round(float((d & ~m).sum()) / d.size, 6)}


if __name__ == '__main__':
    main()
