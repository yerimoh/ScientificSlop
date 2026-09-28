"""Erase only the flagged elements from the original figure instead of redrawing it.

Why this approach. On 0919 we had the editor redraw the figure in TikZ and the gate reverted all four attempts. The reason
differed each time but the root was one: asking it to reproduce 59 elements and the edges between them from text was too much. If the
original is kept and only the expository elements are erased, the rest is preserved pixel for pixel. Since nothing is drawn in by a
generative model and we only erase, nothing new can appear.

Procedure
  1. Take the lines the measurer flagged as they are (the list produced by extra_items.fig_exposition_units)
  2. Ask Qwen2.5-VL for the box where that text is in the image
  3. Paint the box over with the surrounding background colour. PIL only, no generation
  4. Re-transcribe the edited image with the same prompt
  5. Run the measurer's verdict function unchanged to get the new score
  6. Check and record that lines to erase are gone and lines to keep remain

  python3 fig_edit.py <code> [--out DIR]
Needs one GPU.  sr 1 48 --qos=${SLURM_QOS} python3 fig_edit.py FA0002
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
import harness as H          # noqa: E402
import extra_items as X      # noqa: E402

VC = H.ROOT / 'paper/draft_v6/scislopbench/benchA4S/scripts'
TRANSCRIBE_PROMPT = ('Transcribe every piece of text that appears in this figure, verbatim, including numbers, '
                     'symbols, labels inside boxes, axis labels, legend entries and small annotations. Do not '
                     'describe the figure and do not add anything that is not written in it. '
                     'Return JSON: {"lines": ["...", "..."]}')


def _ask(img, prompt: str, max_new: int = 1500) -> str:
    """Same machinery and call style as the transcription pipeline. Qwen2.5-VL-32B, bf16, greedy."""
    import torch
    sys.path.insert(0, str(VC))
    import vision_calls as V
    m, proc = V.model()
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors='pt').to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]


def _json_from(text: str):
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None


def locate(img, targets: list[str]) -> list[dict]:
    """Ask where the text to erase is in the image. Coordinates are returned as fractions between 0 and 1."""
    want = '\n'.join(f'- "{t}"' for t in targets)
    prompt = ('Some text in this figure has to be erased. For each item below, give the rectangle that contains '
              'that text and nothing else. Use fractions of the image width and height, with (0,0) at the top left '
              'and (1,1) at the bottom right. If an item appears more than once, give every occurrence. If an item '
              'is not in the figure, leave it out.\n\n' + want +
              '\n\nReturn JSON only: {"boxes": [{"text": "...", "x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}]}')
    j = _json_from(_ask(img, prompt, 1200)) or {}
    boxes = []
    for b in (j.get('boxes') or []):
        try:
            x0, y0, x1, y1 = (float(b['x0']), float(b['y0']), float(b['x1']), float(b['y1']))
        except Exception:
            continue
        if max(x0, y0, x1, y1) > 1.5:          # when the answer is on a 0..1000 scale
            x0, y0, x1, y1 = x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000
        x0, x1 = sorted((max(0.0, x0), min(1.0, x1)))
        y0, y1 = sorted((max(0.0, y0), min(1.0, y1)))
        if x1 - x0 > 0.001 and y1 - y0 > 0.001 and (x1 - x0) * (y1 - y0) < 0.35:
            boxes.append({'text': b.get('text'), 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    return boxes


def erase(img, boxes: list[dict], pad: int = 2):
    """Paint the box over with the background colour at that spot. Nothing is drawn anew."""
    from PIL import Image, ImageDraw
    import numpy as np
    im = img.convert('RGB').copy()
    a = np.asarray(im)
    d = ImageDraw.Draw(im)
    W, Hh = im.size
    for b in boxes:
        x0, y0 = int(b['x0'] * W) - pad, int(b['y0'] * Hh) - pad
        x1, y1 = int(b['x1'] * W) + pad, int(b['y1'] * Hh) + pad
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(Hh - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        ring = []
        for yy in (max(0, y0 - 3), min(Hh - 1, y1 + 3)):
            ring.append(a[yy, x0:x1])
        for xx in (max(0, x0 - 3), min(W - 1, x1 + 3)):
            ring.append(a[y0:y1, xx])
        ring = [r for r in ring if getattr(r, 'size', 0)]
        bg = tuple(int(v) for v in np.median(np.concatenate(ring, axis=0), axis=0)) if ring else (255, 255, 255)
        d.rectangle([x0, y0, x1, y1], fill=bg)
    return im


def transcribe(img) -> list[str]:
    j = _json_from(_ask(img, TRANSCRIBE_PROMPT, 1500)) or {}
    return [str(x) for x in (j.get('lines') or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from PIL import Image
    out = Path(a.out or (API / 'fig_edit' / a.code))
    out.mkdir(parents=True, exist_ok=True)
    m0 = H.fars_paper_dir(a.code)
    fig = X.method_figure(a.code)
    src = next((p for p in (m0 / (fig or ''), m0 / 'figures' / Path(fig or '').name) if p.exists()), None)
    if src is None:
        print(json.dumps({'code': a.code, 'error': 'figure not found'})); return
    lines = X._transcript(a.code)
    found, _, nk = X._read_kinds(lines, a.code)
    targets = sorted({str(h).strip() for hits in found.values() for h in hits})
    keep = [x.strip() for x in lines if x.strip() and x.strip() not in set(targets)]
    im = Image.open(src).convert('RGB')
    if max(im.size) > 2000:
        sc = 2000.0 / max(im.size)
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    boxes = locate(im, targets)
    edited = erase(im, boxes)
    dst = out / Path(src).name
    edited.save(dst, quality=95)
    new_lines = transcribe(edited)
    nfound, _, _ = X._read_kinds(new_lines, a.code)
    norm = lambda t: re.sub(r'\W+', ' ', str(t)).strip().lower()
    gone = [t for t in targets if not any(norm(t)[:40] and norm(t)[:40] in norm(l) for l in new_lines)]
    lost = [k for k in keep if norm(k) and not any(norm(k)[:30] in norm(l) for l in new_lines)]
    rec = {'code': a.code, 'figure': Path(src).name, 'targets': targets, 'boxes': boxes,
           'score_before': round(len(found) / nk, 4), 'kinds_before': sorted(found),
           'score_after': round(len(nfound) / nk, 4), 'kinds_after': sorted(nfound),
           'removed_as_asked': len(gone) == len(targets), 'targets_still_present': [t for t in targets if t not in gone],
           'kept_lines_total': len(keep), 'kept_lines_missing_after': len(lost),
           'kept_lines_missing_examples': lost[:8],
           'edited_image': str(dst), 'transcript_after': new_lines}
    (out / 'REPORT.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    (out / 'transcript_before.txt').write_text('\n'.join(lines))
    (out / 'transcript_after.txt').write_text('\n'.join(new_lines))
    print(json.dumps({k: v for k, v in rec.items() if k not in ('transcript_after', 'boxes')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
