"""Compare three ways of erasing expository elements from a figure on the same targets and the same image.

  A  mask      deterministic painting over. Fill the target rectangles with the surrounding background colour. No model
  B  lama      mask inpainting. LaMa ONNX, CPU. Outside the target the original is restored
  C  qwen2511  instruction-based editing. No mask; the model is told in a sentence what to erase

All three are judged by the same yardsticks.
  changed-outside ratio   how well the original was preserved. If large, the figure was damaged
  score                   fig_exposition re-measured with the same verdict function
  cleared                 whether the phrase still appears in the re-transcription after the edit
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def preservation(orig: Path, edited: Path, targets, thr: int = 24) -> dict:
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(orig).convert('RGB')).astype(int)
    b = np.array(Image.open(edited).convert('RGB')).astype(int)
    if a.shape != b.shape:
        return {'comparable': False, 'note': 'size changed'}
    d = (np.abs(a - b).sum(2) > thr)
    m = np.zeros(d.shape, bool)
    for x1, y1, x2, y2 in targets:
        m[max(0, y1):y2, max(0, x1):x2] = True
    return {'comparable': True, 'changed_total': int(d.sum()),
            'changed_inside': int((d & m).sum()), 'changed_outside': int((d & ~m).sum()),
            'outside_ratio': round(float((d & ~m).sum()) / d.size, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('src')
    ap.add_argument('outdir')
    ap.add_argument('--boxes-json', required=True)
    ap.add_argument('--drop', action='append', default=[])
    ap.add_argument('--methods', default='mask,lama,qwen2511')
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    located = json.load(open(a.boxes_json))
    targets = [x['box_2d'] for x in located]
    phrases = a.drop or [x['phrase'] for x in located]
    import figure_edit as FE, extra_items as E
    before = E._transcript(a.code)
    fb, _, nk = E._read_kinds(before, a.code)
    rows = []
    for name in [m.strip() for m in a.methods.split(',') if m.strip()]:
        dst = out / f'{name}.jpeg'
        t0 = time.time()
        err = None
        try:
            if name == 'mask':
                FE.erase(Path(a.src), [{'phrase': x['phrase'], 'box_2d': x['box_2d']} for x in located], dst)
            elif name == 'lama':
                import figure_inpaint as FI
                FI.inpaint(Path(a.src), targets, dst)
            elif name == 'qwen2511':
                import figure_edit_gen as FG
                FG.edit(Path(a.src), phrases, dst, model='qwen2511')
            else:
                raise ValueError(name)
        except Exception as e:
            err = f'{type(e).__name__}: {e}'[:300]
        dt = round(time.time() - t0, 1)
        if err or not dst.exists():
            rows.append({'method': name, 'seconds': dt, 'error': err}); continue
        pres = preservation(Path(a.src), dst, targets)
        lines = FE.transcribe(dst)
        (out / f'{name}_lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1))
        fa, _, _ = E._read_kinds(lines, a.code)
        gone = [p for p in phrases if not any(p[:30].lower() in (l or '').lower() for l in lines)]
        rows.append({'method': name, 'seconds': dt, 'preservation': pres,
                     'kinds_before': sorted(fb), 'kinds_after': sorted(fa),
                     'score_before': round(len(fb) / nk, 4), 'score_after': round(len(fa) / nk, 4),
                     'phrases_gone': len(gone), 'phrases_total': len(phrases), 'n_lines': len(lines)})
    (out / 'compare.json').write_text(json.dumps({'code': a.code, 'targets': targets, 'rows': rows},
                                                 indent=1, ensure_ascii=False))
    print(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
