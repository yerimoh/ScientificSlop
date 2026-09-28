"""Instance grounding, figure surface: extract every body figure of the 286 bench165 papers
to PNG. CPU only. Output: results/slop/fig_specimen/figs/<item>__<n>.png + manifest.jsonl"""
import json, os, re, glob, subprocess, sys
ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
sys.path.insert(0, f"{ROOT}/paper/draft_v6/scislopbench/data/scripts")
from slopbench_lib import fars_tex_root  # noqa: E402
OUT = f"{B}/results/slop/fig_specimen/figs"
items = json.load(open(f"{B}/items165.json"))["items"]
man = []
for it in items:
    iid = it["item_id"]
    body = open(f"{B}/views/{iid}/body.tex").read()
    if it["label"] == 1:
        base = os.path.dirname(fars_tex_root(it["pair"]))
    else:
        base = it["tex_dir"]
    names = re.findall(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", body)
    seen = set()
    for n, name in enumerate(names):
        name = name.strip()
        if name in seen: continue
        seen.add(name)
        cands = []
        for ext in ("", ".pdf", ".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            cands += glob.glob(os.path.join(base, name + ext)) + glob.glob(os.path.join(base, "**", os.path.basename(name) + ext), recursive=True)
        src = next((c for c in cands if os.path.isfile(c)), None)
        if not src: 
            man.append({"item_id": iid, "fig": name, "png": None, "err": "not_found"}); continue
        dst = f"{OUT}/{iid}__{n:02d}.png"
        if not os.path.exists(dst):
            try:
                if src.lower().endswith(".pdf"):
                    subprocess.run(["pdftoppm", "-png", "-r", "130", "-f", "1", "-l", "1",
                                    "-singlefile", src, dst[:-4]], check=True, timeout=60,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    from PIL import Image
                    Image.open(src).convert("RGB").save(dst)
            except Exception as e:
                man.append({"item_id": iid, "fig": name, "png": None, "err": repr(e)[:60]}); continue
        if os.path.exists(dst):
            man.append({"item_id": iid, "fig": name, "png": os.path.basename(dst)})
with open(f"{B}/results/slop/fig_specimen/manifest.jsonl", "w") as w:
    for m in man: w.write(json.dumps(m) + "\n")
ok = sum(1 for m in man if m.get("png"))
print("figures extracted:", ok, "of", len(man), "refs; papers:", len({m['item_id'] for m in man}))
