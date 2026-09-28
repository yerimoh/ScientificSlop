"""Deterministically cuts text lines out of a diagram and asks the VLM only which line is the phrase to erase.

Why not ask for coordinates. On 0919 Qwen2.5-VL was asked for boxes directly three times and all three were off by tens of pixels.
Once it erased the wrong spot and clipped a box of the mechanism. Coordinate regression is a weak point of this model, whereas
"transcribe the text of this chunk verbatim" is a strong one. So the cutting is done by pixels and only the reading is left to the model.

Procedure
  1. Pixels that differ from the background colour count as ink.
  2. A horizontal projection splits at empty rows into bands; a vertical projection inside each band splits into chunks.
  3. Only text-like chunks are kept (filtered by height and aspect ratio).
  4. The chunks are stacked with index numbers on one sheet, sent to the VLM once, and the text per index is returned.
  5. The phrase to erase is string-matched against the text and that chunk's rectangle is returned.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

SHEET_PROMPT = (
    'This image is a contact sheet. It stacks small crops taken from a figure, one per row, and each row has an '
    'index number printed at its left in red. For every row, transcribe the text of the crop verbatim.\n\n'
    'Return JSON only: {"rows": [{"index": 0, "text": "..."}, ...]}. Use an empty string for a row with no readable '
    'text. Do not describe anything and do not merge rows.')


def dark_mask(a, thr: int = 150):
    """Pixels to treat as text. Diagrams mix a white background with light grey panels, so one background colour cannot separate
    them. Text is clearly darker than both, so we separate by luminance. Box borders and arrows are dark too, but the size
    conditions later filter them out."""
    import numpy as np
    lum = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
    return (lum < thr)


def _runs(mask1d, min_gap: int = 2):
    out, start = [], None
    gap = 0
    for i, v in enumerate(mask1d):
        if v:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    out.append((start, i - gap + 1)); start = None; gap = 0
    if start is not None:
        out.append((start, len(mask1d)))
    return out


def candidates(image_path: Path, max_regions: int = 90) -> list[tuple]:
    """List of rectangles of chunks that look like text lines. Characters are joined horizontally into one line, then blobs are counted."""
    import cv2
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(image_path).convert('RGB'))
    H, W = a.shape[:2]
    dark = dark_mask(a).astype(np.uint8)
    # Fill the gaps between characters to join them into one line. No vertical joining, so lines do not merge.
    joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 8 or h > 46:                 # height of a text line
            continue
        if w < 28 or w / h < 1.6:           # too short means a symbol or an arrowhead
            continue
        if area / float(w * h) > 0.75:      # a solid blob is a filled box
            continue
        if w > 0.92 * W:                    # a line spanning the diagram
            continue
        out.append((int(x), int(y), int(x + w), int(y + h)))
    out.sort(key=lambda b: (b[1], b[0]))
    return out[:max_regions]


def contact_sheet(image_path: Path, boxes: list[tuple], out_path: Path, row_h: int = 34, pad: int = 6):
    from PIL import Image, ImageDraw
    src = Image.open(image_path).convert('RGB')
    rows = []
    for b in boxes:
        c = src.crop(b)
        sc = row_h / max(1, c.height)
        c = c.resize((max(1, int(c.width * sc)), row_h), Image.LANCZOS)
        rows.append(c)
    Wm = max([r.width for r in rows] + [1]) + 70
    Hm = sum(r.height + pad for r in rows) + pad
    sheet = Image.new('RGB', (Wm, Hm), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y = pad
    for i, r in enumerate(rows):
        d.text((6, y + row_h // 3), str(i), fill=(220, 0, 0))
        sheet.paste(r, (60, y))
        y += r.height + pad
    sheet.save(out_path)
    return sheet


# ------------------------------------------------------------------ reader model
# 0919 evening. Reading the sheet and chunks does not need Qwen2.5-VL-32B. That model demands two 48GB cards, which tied
# rounds to the GPU queue. Having Haiku open the image with the Claude Code CLI's Read tool finishes in seconds and the result
# format is the same. The measurer's original transcription still uses the Qwen cache, so the scoring baseline is unchanged. What
# changes is the eye that confirms "which chunk is the phrase to erase", and the rule that no confirmation means no erasing stands.
READER = os.environ.get('SH_FIG_READER', 'claude')            # claude | qwen
READER_MODEL = os.environ.get('SH_FIG_READER_MODEL', 'claude-haiku-4-5-20251001')


def _ask_claude(img, prompt: str, workdir: Path | None = None) -> str:
    """Saves the image to a file and has the CLI open it with Read. Writing and the shell are blocked."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'temp' / 'code'))
    import harness as H
    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix='figread_', dir=str(workdir) if workdir else None))
    fp = d / 'crop.png'
    img.save(fp)
    full = (f'Open the image file crop.png in this directory with the Read tool and look at it. {prompt} '
            f'Return the JSON only, no prose.')
    r = H.run_cli_json(['-p', full, '--model', READER_MODEL, '--permission-mode', 'acceptEdits', '--add-dir', str(d),
                        '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit,Write', '--restricted'], d, timeout=300)
    return r.get('result') or ''


def _ask(img, prompt: str, workdir: Path | None = None) -> str:
    if READER == 'qwen':
        import figure_edit as FE
        return FE._ask(img, prompt)
    return _ask_claude(img, prompt, workdir)


def read_sheet(sheet_img, workdir: Path | None = None) -> dict:
    out = _ask(sheet_img, SHEET_PROMPT, workdir)
    t = re.sub(r'^```(?:json)?|```$', '', (out or '').strip(), flags=re.M)
    m = re.search(r'\{.*\}', t, re.S)
    if not m:
        return {}
    try:
        j = json.loads(m.group(0))
    except Exception:
        return {}
    return {int(r['index']): str(r.get('text') or '') for r in j.get('rows', []) if 'index' in r}


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def _score(a: str, b: str) -> float:
    import difflib
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return min(len(na), len(nb)) / max(len(na), len(nb))
    return difflib.SequenceMatcher(None, na, nb).ratio()


CROP_PROMPT = ('Transcribe the text in this image verbatim. Return JSON only: {"text": "..."}. '
               'Do not describe anything.')


def _read_crop(img, workdir: Path | None = None) -> str:
    out = _ask(img, CROP_PROMPT, workdir)
    t = re.sub(r'^```(?:json)?|```$', '', (out or '').strip(), flags=re.M)
    m = re.search(r'\{.*\}', t, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get('text') or '')
        except Exception:
            pass
    return t.strip()[:200]


CACHE = Path(__file__).resolve().parent.parent / 'figure_band_cache'


def _key(image_path: Path) -> str:
    import hashlib
    return hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:16]


def locate_phrases(image_path: Path, phrases: list[str], workdir: Path, verify: bool = True) -> list[dict]:
    """Returns the rectangle of the matching chunk for each phrase.

    The model is not asked to read the numbers. On 0919 it read the sheet's red numbers off by one and picked the wrong chunk. Chunks
    are sorted top to bottom, so we match by the order of the returned lines and re-read the chosen chunk to confirm. If that
    fails we look at the neighbours above and below. A phrase that still fails confirmation is not erased. Not erasing beats erasing wrongly."""
    from PIL import Image
    workdir.mkdir(parents=True, exist_ok=True)
    boxes = candidates(image_path)
    if not boxes:
        return []
    # The same figure stays the same across rounds. Loading the 32B transcription model from disk each time is almost the whole
    # cost of this step, so the sheet reading and chunk confirmations are cached by image hash.
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = CACHE / f'{_key(image_path)}.json'
    cached = {}
    if ck.exists():
        try:
            cached = json.loads(ck.read_text())
        except Exception:
            cached = {}
    sheet_path = workdir / 'contact_sheet.png'
    if cached.get('texts') and [list(b) for b in (cached.get('boxes') or [])] == [list(b) for b in boxes]:
        texts_by_index = {i: t for i, t in enumerate(cached['texts'])}
    else:
        sheet = contact_sheet(image_path, boxes, sheet_path)
        texts_by_index = read_sheet(sheet, workdir)
    ordered = [texts_by_index.get(i, '') for i in range(len(boxes))]
    if len(texts_by_index) == len(boxes):
        ordered = [texts_by_index[i] for i in sorted(texts_by_index)]
    (workdir / 'sheet_texts.json').write_text(json.dumps(
        {'boxes': boxes, 'texts': ordered}, indent=1, ensure_ascii=False))
    src = Image.open(image_path).convert('RGB')
    out = []
    for p in phrases:
        ranked = sorted(range(len(boxes)), key=lambda i: -_score(ordered[i] if i < len(ordered) else '', p))
        # The read lines are sometimes shifted by one relative to the chunks. If the model inserts one empty line, everything after it shifts.
        # So not only the best match but also its neighbours are actually opened and checked.
        order = []
        for i in ranked[:3]:
            if _score(ordered[i] if i < len(ordered) else '', p) < 0.35:
                break
            for j in (i, i + 1, i - 1):
                if 0 <= j < len(boxes) and j not in order:
                    order.append(j)
        chosen = None
        for j in order:
            if not verify:
                chosen = (j, _score(ordered[j], p), ordered[j]); break
            ckey = f'crop{j}'
            got = (cached.get('crops') or {}).get(ckey)
            if got is None:
                got = _read_crop(src.crop(boxes[j]), workdir)
                cached.setdefault('crops', {})[ckey] = got
            sc = _score(got, p)
            # 0920. On FA0050 'Step 3: Math-Verify' matched the 'Math-Verify' chunk (0.67) and only the mechanism name was erased. Raise the confirmation threshold.
            if sc >= float(os.environ.get('SH_FIG_MATCH_MIN', '0.8')):
                chosen = (j, sc, got); break
        if chosen:
            i, sc, got = chosen
            out.append({'phrase': p, 'box_2d': list(boxes[i]), 'row': i, 'row_text': got,
                        'match': round(sc, 3), 'verified': verify})
    (workdir / 'located.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    cached.update({'boxes': [list(b) for b in boxes], 'texts': ordered})
    ck.write_text(json.dumps(cached, ensure_ascii=False))
    return out
