"""Erase expository elements with instruction-based image editing and verify at the pixel level that the original was really preserved.

Why verification is the core. On 0917 this project already saw a generative model touch a figure, draw lines that were not there
and misspell text. The moment a figure is handed to a generative model, fabrication becomes possible, so editing is allowed only when
"nothing outside the target changed" can be proven. The proof is a pixel comparison.

  python3 figure_edit_gen.py <code> <in> <out> --drop "phrase" [--drop ...] [--model qwen|flux]

Verdict
  changed_outside_ratio  fraction of pixels changed outside the target rectangles. If large, the model touched other areas
  target_cleared         whether the text inside the target really disappeared (checked by re-transcribing after the edit)
  verdict                accept or reject. On reject, fall back to deterministic painting over
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INSTRUCTION = ('Remove the text "{phrase}" from this figure. Erase only those characters and the empty banner or '
               'box that holds them if it holds nothing else. Leave every other part of the image exactly as it is. '
               'Do not move, redraw, recolour or relabel any box, arrow, line or other text. Do not add anything. '
               'The result must be the same diagram with that text gone.')


def edit(image_path: Path, phrases: list[str], out_path: Path, model: str = 'qwen2511', steps: int = 30,
         true_cfg: float = 4.0, seed: int = 914):
    import torch
    from PIL import Image
    im = Image.open(image_path).convert('RGB')
    instr = ' '.join(INSTRUCTION.format(phrase=p) for p in phrases)
    if model == 'flux':
        from diffusers import FluxKontextPipeline
        pipe = FluxKontextPipeline.from_pretrained('black-forest-labs/FLUX.1-Kontext-dev',
                                                   torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        out = pipe(image=im, prompt=instr, guidance_scale=2.5, num_inference_steps=steps,
                   generator=torch.Generator().manual_seed(seed)).images[0]
    else:
        from diffusers import QwenImageEditPlusPipeline
        repo = 'Qwen/Qwen-Image-Edit-2511' if model == 'qwen2511' else 'Qwen/Qwen-Image-Edit-2509'
        n = torch.cuda.device_count()
        if n >= 2:
            # Sharding each component separately mismatches devices. Let balanced placement handle the whole pipeline.
            pipe = QwenImageEditPlusPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                             device_map='balanced')
        else:
            pipe = QwenImageEditPlusPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
        # The VAE does not fit whole on a single 48GB card. Encode in tiles and shrink the side if needed.
        for fn in ('enable_vae_tiling', 'enable_vae_slicing'):
            if hasattr(pipe, fn):
                getattr(pipe, fn)()
            elif hasattr(getattr(pipe, 'vae', None), fn):
                getattr(pipe.vae, fn)()
        work = im
        cap = int(os.environ.get('SH_EDIT_MAXSIDE', '1024'))
        if max(work.size) > cap:
            sc = cap / max(work.size)
            work = work.resize((max(8, int(work.width * sc) // 8 * 8),
                                max(8, int(work.height * sc) // 8 * 8)), Image.LANCZOS)
        out = pipe(image=[work], prompt=instr, num_inference_steps=steps, true_cfg_scale=true_cfg,
                   negative_prompt=' ', generator=torch.Generator().manual_seed(seed)).images[0]
    if out.size != im.size:
        out = out.resize(im.size, Image.LANCZOS)
    out.save(out_path, quality=95)
    return out_path


def preservation(orig: Path, edited: Path, targets: list[list[int]], thr: int = 24) -> dict:
    """How much changed outside the target. This must be small for the original to count as preserved."""
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(orig).convert('RGB')).astype(int)
    b = np.array(Image.open(edited).convert('RGB')).astype(int)
    if a.shape != b.shape:
        return {'comparable': False, 'note': 'the edit changed the image size'}
    diff = (np.abs(a - b).sum(2) > thr)
    mask = np.zeros(diff.shape, bool)
    for x1, y1, x2, y2 in targets:
        mask[max(0, y1):y2, max(0, x1):x2] = True
    outside = diff & ~mask
    inside = diff & mask
    ys, xs = np.where(outside)
    return {'comparable': True,
            'changed_total': int(diff.sum()),
            'changed_inside_target': int(inside.sum()),
            'changed_outside_target': int(outside.sum()),
            'changed_outside_ratio': round(float(outside.sum()) / diff.size, 6),
            'target_pixels': int(mask.sum()),
            'inside_cleared_ratio': round(float(inside.sum()) / max(1, int(mask.sum())), 4),
            'outside_bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--drop', action='append', default=[])
    ap.add_argument('--model', default='qwen2511', choices=['qwen2511', 'qwen', 'flux'])
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--report', default='')
    a = ap.parse_args()
    import figure_bands as B
    wd = Path(a.dst).parent / '_bands_gen'
    located = B.locate_phrases(Path(a.src), a.drop, wd)
    targets = [l['box_2d'] for l in located]
    edit(Path(a.src), a.drop, Path(a.dst), model=a.model, steps=a.steps)
    pres = preservation(Path(a.src), Path(a.dst), targets)
    import figure_edit as F
    lines_after = F.transcribe(Path(a.dst))
    gone = [p for p in a.drop if not any(p[:30].lower() in (l or '').lower() for l in lines_after)]
    rec = {'model': a.model, 'requested': a.drop, 'located': located, 'preservation': pres,
           'phrases_gone': gone, 'phrases_left': [p for p in a.drop if p not in gone],
           'lines_after': lines_after}
    rec['verdict'] = ('accept' if pres.get('comparable') and pres['changed_outside_ratio'] < 0.002
                      and len(gone) == len(a.drop) else 'reject')
    if a.report:
        Path(a.report).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rec.items() if k != 'lines_after'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
