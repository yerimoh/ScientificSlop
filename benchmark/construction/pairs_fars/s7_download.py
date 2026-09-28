"""Step 7: download e-prints for cache/download_queue.json only (final primary/alternate
candidates that are not in the human_refs corpus). 3 s between downloads.

Saves original directory structure under pairs165_0911/eprints_new/<id>/, then rerun
s4_cand_metrics.py and s6_rank_assign.py.
"""
import os, io, json, time, tarfile, gzip, urllib.request

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911"

queue = json.load(open(f"{OUTD}/cache/download_queue.json"))
for aid in queue:
    dst = f"{OUTD}/eprints_new/{aid}"
    if os.path.isdir(dst) and os.listdir(dst):
        continue
    url = f"https://arxiv.org/e-print/{aid}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scislopbench-pairs/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except Exception as e:
        print(aid, "FAIL", type(e).__name__, e)
        time.sleep(5)
        continue
    os.makedirs(dst, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            safe = [m for m in tf.getmembers()
                    if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in m.name]
            tf.extractall(dst, members=safe)
        print(aid, "tar ok", len(safe), "files")
    except tarfile.ReadError:
        try:
            txt = gzip.decompress(raw)
        except OSError:
            txt = raw
        open(f"{dst}/main.tex", "wb").write(txt)
        print(aid, "single-file tex")
    time.sleep(3)
print("done", len(queue))
