"""fig_exposition inputs for benchA4S, step 2 of 2: keep only genuine method figures.

pairs165 had a person decide, from each paper's caption list, which figure is the method figure and
which paper draws none. The A4S set is too large for that, so the same decision is made by the
vision model both sides are transcribed with, from the crop and its caption, with one yes/no
question and no other context. The first candidate the model accepts, in figure order, becomes that
paper's method figure; a paper the model accepts nothing from is "none" and is NA for the item,
never clean.

The chosen crops are copied to figexp/data/crops under the names the item's reader expects, so
nothing downstream has to know that candidates were considered at all.

Writes figexp/data/{gate.json, crops/, manifest.json, hu_figure.json}.
"""
import json, os, shutil, sys

import torch
from PIL import Image

B = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/benchA4S"
D = f"{B}/figexp/data"
sys.path.insert(0, f"{B}/scripts")
from vision_calls import model  # noqa: E402

Q = ("You are shown one figure cropped from a research paper, and its caption.\n\n"
     "Caption: {cap}\n\n"
     "Question: does this figure diagram how the paper's own method, system or pipeline works, that "
     "is, does it draw the parts of the approach and how they connect or how data moves through "
     "them?\n\n"
     "Answer no if it is a plot of results, a table, an example of the data, a qualitative output, a "
     "photograph or apparatus picture, a map, a molecular or anatomical rendering shown as data, a "
     "screenshot, or a figure about prior work only.\n\n"
     "Answer with exactly one word, yes or no.")


def ask(m, proc, png, cap):
    img = Image.open(png).convert("RGB")
    w, h = img.size
    if max(w, h) > 1400:
        s = 1400 / max(w, h)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))))
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q.format(cap=cap[:300])}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=4, do_sample=False)
    ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return ans.strip().lower().startswith("y"), ans.strip()


def main():
    cands = json.load(open(f"{D}/candidates.json"))
    cache_path = f"{D}/gate.json"
    gate = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    m, proc = model()
    n = 0
    for code, v in cands.items():
        for side in ("ai", "hu"):
            for c in v[side]:
                k = c["png"]
                if k in gate:
                    continue
                try:
                    ok, raw = ask(m, proc, k, c["caption"])
                except Exception as e:
                    ok, raw = False, f"error {e}"
                gate[k] = {"method": ok, "raw": raw}
                n += 1
                if n % 25 == 0:
                    json.dump(gate, open(cache_path, "w"), indent=1)
                    print(f"  gated {n}", flush=True)
    json.dump(gate, open(cache_path, "w"), indent=1)

    crops = f"{D}/crops"
    os.makedirs(crops, exist_ok=True)
    man, hu_fig = [], {}
    for code, v in cands.items():
        ai = next((c for c in v["ai"] if gate.get(c["png"], {}).get("method")), None)
        hu = next((c for c in v["hu"] if gate.get(c["png"], {}).get("method")), None)
        ai_png = hu_png = None
        if ai:
            ai_png = f"{crops}/AI_{code}.png"
            shutil.copyfile(ai["png"], ai_png)
        if hu:
            hu_png = f"{crops}/HU_{v['arxiv']}.png"
            shutil.copyfile(hu["png"], hu_png)
        man.append({"code": code, "arxiv": v["arxiv"],
                    "ai_png": ai_png, "ai_caption": ai["caption"] if ai else None,
                    "hu_png": hu_png, "hu_caption": hu["caption"] if hu else None})
        hu_fig[v["arxiv"]] = f"fig{hu['fig_no']}" if hu else "none"
    json.dump(man, open(f"{D}/manifest.json", "w"), indent=1)
    json.dump(hu_fig, open(f"{D}/hu_figure.json", "w"), indent=1)
    print(f"papers {len(man)} | AI method figure {sum(1 for x in man if x['ai_png'])} | "
          f"HU method figure {sum(1 for x in man if x['hu_png'])}")


if __name__ == "__main__":
    main()
