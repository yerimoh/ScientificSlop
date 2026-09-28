"""Whole-figure transcription with Qwen2.5-VL-32B-Instruct, one call per image, same prompt for the AI
PNGs and the human crops. Writes <out>/transcripts.jsonl (resumable)."""
import json, os, sys, glob
from PIL import Image
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/fig_graph/roi_extract")
import vision_calls as V

S = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/candidte/_census_0914/pairs165_full/data"
OUT = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/candidte/_census_0914/pairs165_full"
os.makedirs(OUT, exist_ok=True)
PROMPT = ("Transcribe every piece of text that appears in this figure, verbatim, including numbers, symbols, "
          "labels inside boxes, axis labels, legend entries and small annotations. Do not describe the figure and do "
          "not add anything that is not written in it. Return JSON: {\"lines\": [\"...\", \"...\"]}")


def load(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 2000:
        sc = 2000.0 / max(im.size)
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    return im


def main():
    man = json.load(open(f"{S}/pairs165_manifest.json"))
    hu = {r["arxiv"]: r for r in json.load(open(f"{S}/pairs165_crops/hu_figs.json"))}
    side = sys.argv[1] if len(sys.argv) > 1 else "both"
    jobs = []
    for r in man:
        if r["ai_png"] and side in ("both", "AI"):
            jobs.append((f"AI_{r['code']}", r["ai_png"]))
        h = hu.get(r["arxiv"])
        if h and h.get("status") == "ok" and h.get("verified", True) and side in ("both", "HU"):
            jobs.append((f"HU_{r['arxiv']}", h["crop"]))
    done = set()
    outp = f"{OUT}/transcripts.jsonl"
    if os.path.exists(outp):
        done = {json.loads(l)["key"] for l in open(outp)}
    with open(outp, "a") as fo:
        for key, path in jobs:
            if key in done:
                continue
            try:
                res = V.ask(load(path), PROMPT, max_new=900)
            except Exception as e:
                res = {"error": str(e)}
            fo.write(json.dumps({"key": key, "path": path, "result": res}, ensure_ascii=False) + "\n")
            fo.flush()
            print(key, str(res)[:120], flush=True)


if __name__ == "__main__":
    main()
