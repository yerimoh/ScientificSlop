"""Run fig_exposition over benchA4S by repointing its manifest, crops and figure table.

The item's own inputs are the FARS census tables. benchA4S supplies its own manifest and crops
(figexp/data), localised the same way on both sides, so the module constants are rebound before
main() runs. The transcription cache stays in the item's own directory, so a figure read once is
never read twice.
"""
import argparse, json, os, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
B = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S"
D = f"{B}/figexp/data"

ap = argparse.ArgumentParser()
ap.add_argument("--transcribe", action="store_true")
ap.add_argument("--corpus", default="all")
a = ap.parse_args()

cdir = f"{SLOP}/Artifacts/fig_exposition/code"
sys.path.insert(0, f"{SLOP}/_common")
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
sys.path.insert(0, cdir)
os.chdir(cdir)
import measure  # noqa: E402

measure.MANIFEST = f"{D}/manifest.json"
measure.CROPS = f"{D}/cand"
measure.HU_FIGURE = f"{D}/hu_figure.json"
measure.RESULTS = f"{B}/results/slop/fig_exposition"
# CACHES was built from the item's own RESULTS at import time, so rebinding RESULTS alone would
# leave the transcripts this bench writes unreadable on the next pass.
measure.CACHES = [f"{B}/figexp/transcripts.jsonl",
                  f"{measure.RESULTS}/transcripts.jsonl"] + list(measure.CACHES)


def figures_a4s():
    """Both sides read out of this bench's manifest.

    The item's own figures() finds the human crop by name under CROPS, which assumes the census
    layout of one crop per paper. Here the gate picked one candidate out of several per paper, so
    the chosen file is what the manifest records and the name carries no meaning.
    """
    out = []
    for m in json.load(open(measure.MANIFEST)):
        if m.get("ai_png") and os.path.exists(m["ai_png"]):
            out.append({"corpus": "AI", "id": m["code"], "key": f"AI_{m['code']}",
                        "pair": m["arxiv"], "path": m["ai_png"]})
        if m.get("hu_png") and os.path.exists(m["hu_png"]):
            out.append({"corpus": "HU", "id": m["arxiv"], "key": f"HU_{m['arxiv']}",
                        "pair": m["code"], "path": m["hu_png"]})
    return out


measure.figures = figures_a4s

# The item's experimental-content verdicts are the FARS census's, so on this bench every figure
# would score zero on the one kind a pattern cannot decide. benchA4S keeps its own verdict file,
# written under the same rule, and the two are merged so nothing the item already decided is lost.
sys.path.insert(0, f"{B}/figexp")
import leak_verdicts_a4s as LA  # noqa: E402
measure.LEAK.CLEAR = {**measure.LEAK.CLEAR, **LA.CLEAR}
measure.LEAK.CLEAR_AI, measure.LEAK.CLEAR_HU = LA.CLEAR_AI, LA.CLEAR_HU
measure.LEAK.BORDER_AI, measure.LEAK.FALSE_AI = LA.BORDER_AI, LA.FALSE_AI
os.makedirs(measure.RESULTS, exist_ok=True)
sys.argv = ["measure.py", "--corpus", a.corpus] + (["--transcribe"] if a.transcribe else [])
measure.main()
print("WROTE", measure.RESULTS)
