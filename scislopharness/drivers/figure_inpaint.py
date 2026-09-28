"""Erase expository elements with mask-based inpainting. Uses LaMa (ONNX, CPU).

Why this. All we do is "erase text and fill in the background where it was"; no generative reconstruction is needed.
LaMa is a small model trained for exactly that, so it barely damages the original and runs without a GPU. Large instruction-based
editing models risk touching pixels outside the target and required several 48GB cards. The mask is already obtained deterministically
(figure_bands), so there is no need to ask a model where to erase.

Difference from simply painting over with the background colour: when text sits on a grey panel or box, painting over leaves a rectangular mark.
Inpainting continues the surrounding texture and leaves no mark.

  python3 figure_inpaint.py <in> <out> --box x1 y1 x2 y2 [--box ...] [--pad 3]
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

MODEL_REPO = 'Carve/LaMa-ONNX'
MODEL_FILE = os.environ.get('SH_LAMA_FILE', 'lama_fp32.onnx')
SIDE = 512     # LaMa ONNX takes a square 512 input


def _model_path() -> str:
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, MODEL_FILE)


def inpaint(src: Path, boxes: list[list[int]], dst: Path, pad: int = 3, feather: int = 2) -> dict:
    """Erase the inside of the boxes and fill with the surrounding texture. Pixels outside the boxes are restored from the original."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    used = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(W, int(x2) + pad), min(H, int(y2) + pad)
        mask[y1:y2, x1:x2] = 255
        used.append([x1, y1, x2, y2])
    sess = ort.InferenceSession(_model_path(), providers=['CPUExecutionProvider'])
    iname = [i.name for i in sess.get_inputs()]
    im_s = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    mk_s = cv2.resize(mask, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    x = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype('float32') / 255.0
    m = (mk_s[None, None] > 127).astype('float32')
    out = sess.run(None, {iname[0]: x, iname[1]: m})[0]
    out = out[0].transpose(1, 2, 0)
    if out.max() <= 1.01:
        out = out * 255.0
    out = cv2.cvtColor(np.clip(out, 0, 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    # Outside the boxes the original is used as is. Even if the model changes something else, it does not enter the result.
    soft = cv2.GaussianBlur(mask.astype('float32') / 255.0, (0, 0), feather) if feather else mask / 255.0
    soft = np.clip(soft, 0, 1)[:, :, None]
    merged = (out.astype('float32') * soft + img.astype('float32') * (1 - soft)).astype('uint8')
    cv2.imwrite(str(dst), merged, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {'boxes_used': used, 'size': [W, H], 'model': f'{MODEL_REPO}/{MODEL_FILE}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--box', action='append', nargs=4, type=int, default=[])
    ap.add_argument('--boxes-json', default='')
    ap.add_argument('--pad', type=int, default=3)
    a = ap.parse_args()
    boxes = [list(b) for b in a.box]
    if a.boxes_json:
        boxes += [x['box_2d'] for x in json.load(open(a.boxes_json))]
    print(json.dumps(inpaint(Path(a.src), boxes, Path(a.dst), a.pad), ensure_ascii=False))


if __name__ == '__main__':
    main()
