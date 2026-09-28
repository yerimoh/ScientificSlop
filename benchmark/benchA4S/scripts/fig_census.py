"""A census of figure devices that fig_exposition does not define, asked of every method figure of
both benches with one question set.

Why. fig_exposition reads a transcript, so a device that lives in the drawing rather than in its
words cannot reach it. A colour-swatch legend whose entries are bare category words is the case
that started this; the transcript keeps "Agents, Tools, Misc" and loses the swatches, and the
notation_key pattern needs a Legend header or a colour-equals-role line. The questions below are
the candidates that reading the figures suggested, and they are put to the FARS pairs as well, so a
device can be told apart as a property of AI figures or as a signature of one pipeline.

Nothing here is an item. A candidate becomes one only if it separates on both benches and is not a
rendering artefact of how the two sides were cropped.

Writes figexp/census/<bench>.json.
"""
import argparse, json, os, sys

import torch
from PIL import Image

B = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/benchA4S"
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
CENSUS = f"{SLOP}/Artifacts/candidte/_census_0914"
sys.path.insert(0, f"{B}/scripts")
from vision_calls import model  # noqa: E402

Q = {
 "legend_any": "Does the figure contain a key that maps a colour, a shape or a line style to a "
               "category or role, in any form, including a small block of coloured swatches each "
               "labelled with a word, whether or not the word Legend appears?",
 "icons": "Does the figure use pictorial icons as decoration on its nodes, for example a robot, a "
          "person, a database cylinder, a cloud, a gear or a document sheet?",
 "sketch": "Is the figure drawn in a hand-drawn or sketch style, with wobbly lines, as produced by "
           "a whiteboard tool, rather than with straight vector lines?",
 "uniform_boxes": "Are most of the boxes the same size and filled from the same small set of pale "
                  "colours, so that the boxes look laid out automatically rather than drawn to fit "
                  "their content?",
 "inner_title": "Is there a title line drawn inside the figure area, above or beside the diagram, "
                "that names the figure or the system?",
 "single_chain": "Is the whole diagram one unbranched chain of boxes running left to right or top "
                 "to bottom, with no branch, no loop and no parallel path?",
 "misspelling": "Is any English word inside the figure misspelled, or is any word in a language "
                "other than English?",
 # Presentation-software chrome. A4S0258 carried all four at once, and a figure drawn in a vector
 # editor or in TikZ carries none of them.
 "gradient_shadow": "Are the boxes filled with a gradient, or do they cast a drop shadow, or are "
                    "they drawn with a bevelled or three-dimensional edge?",
 "block_arrows": "Are the arrows thick block arrows or three-dimensional arrows, rather than thin "
                 "lines ending in a small head?",
 "outer_frame": "Is the whole figure enclosed in a drawn border or frame around its edge?",
 "clipart": "Does the figure contain a stock clipart picture, for example a funnel, a gauge, a "
            "stopwatch, a lightbulb, a magnifying glass or a trophy?",
 # Round two, written after opening a dozen A4S method figures rather than from a guess.
 "garbled_text": "Does the figure contain a word that is not a real word, a run of letters that "
                 "looks like text but is not, or a word with letters visibly doubled or dropped, "
                 "of the kind an image generator produces?",
 "text_collision": "Does any text in the figure overlap other text or a box edge, or is any text "
                   "cut off, so that it cannot be read?",
 "bulleted_cards": "Are there three or more boxes that each carry a bold heading with a bulleted "
                   "list under it, so that the figure reads as a set of slide cards?",
 "decorative_shapes": "Do the nodes use several different shapes or fill colours with nothing in "
                      "the figure explaining what the shapes or colours mean?",
 "whole_paper": "Does the figure summarise the whole paper, showing the data, the method and the "
                "results or conclusions together, rather than diagramming the method alone?",
 # Round three, written after putting the human method figures beside the A4S ones. The human
 # figures draw the thing itself, a crystal, a spectrum, a cloud of points, a bar chart, where the
 # A4S figures name it in a box.
 "depicts_data": "Does the figure draw any actual object of the work, for example a small plot, a "
                 "curve, a matrix or heatmap, a sample image, a molecule or structure, or a piece "
                 "of real input text, rather than only naming things inside boxes?",
 "labeled_arrows": "Is any arrow in the figure labelled with what passes along it or what happens "
                   "at that step?",
 "repetition_glyph": "Does the figure use an ellipsis, stacked copies of a shape, or a similar "
                     "device to stand for many items of the same kind?",
 "equation_inside": "Is a mathematical formula written inside the figure?"
}
PROMPT = ("You are shown one figure from a research paper.\n\n{q}\n\n"
          "Answer with exactly one word, yes or no.")


def ask(m, proc, img, q):
    msg = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": PROMPT.format(q=q)}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=4, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0].strip().lower().startswith("y")


def load(p):
    img = Image.open(p).convert("RGB")
    w, h = img.size
    if max(w, h) > 1400:
        s = 1400 / max(w, h)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))))
    return img


def figures(bench):
    if bench == "benchA4S":
        out = []
        for m in json.load(open(f"{B}/figexp/data/manifest.json")):
            if m.get("ai_png"):
                out.append(("AI", m["code"], m["ai_png"]))
            if m.get("hu_png"):
                out.append(("HU", m["arxiv"], m["hu_png"]))
        return out
    crops = f"{CENSUS}/pairs165_full/data/pairs165_crops"
    hu_fig = json.load(open(f"{CENSUS}/pairs165_hu_overrides.json"))
    out = []
    for m in json.load(open(f"{CENSUS}/pairs165_full/data/pairs165_manifest.json")):
        if m.get("ai_png") and os.path.exists(m["ai_png"]):
            out.append(("AI", m["code"], m["ai_png"]))
        c = os.path.join(crops, f"HU_{m['arxiv']}.png")
        if hu_fig.get(m["arxiv"]) != "none" and os.path.exists(c):
            out.append(("HU", m["arxiv"], c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchA4S", choices=["benchA4S", "bench165"])
    a = ap.parse_args()
    os.makedirs(f"{B}/figexp/census", exist_ok=True)
    path = f"{B}/figexp/census/{a.bench}.json"
    res = json.load(open(path)) if os.path.isfile(path) else {}
    m, proc = model()
    figs = figures(a.bench)
    print(a.bench, "figures", len(figs), flush=True)
    for i, (corpus, pid, png) in enumerate(figs):
        key = f"{corpus}_{pid}"
        res.setdefault(key, {"corpus": corpus, "id": pid})
        if all(q in res[key] for q in Q):
            continue
        try:
            img = load(png)
        except Exception as e:
            res[key]["error"] = str(e)[:120]
            continue
        for name, q in Q.items():
            if name in res[key]:
                continue
            res[key][name] = ask(m, proc, img, q)
        if i % 20 == 0:
            json.dump(res, open(path, "w"), indent=1)
            print(f"  [{i+1}/{len(figs)}] {key}", flush=True)
    json.dump(res, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
