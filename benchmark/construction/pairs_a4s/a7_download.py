"""Step 7 (Agents4Science): fetch e-prints listed in cache/download_queue.json, original
directory structure, 3 s apart. Then rerun a4_cand_metrics.py and a6_rank_assign.py.
"""
import os, io, sys, json, time, tarfile, gzip, urllib.request

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"

queue = json.load(open(f"{OUTD}/cache/download_queue.json"))
ok = 0
for n, aid in enumerate(queue):
    dst = f"{OUTD}/eprints_new/{aid}"
    if os.path.isdir(dst) and os.listdir(dst):
        continue
    try:
        req = urllib.request.Request(f"https://export.arxiv.org/e-print/{aid}",
                                     headers={"User-Agent": "scislopbench-a4s/0.1 (user@example.org)"})
        raw = urllib.request.urlopen(req, timeout=90).read()
    except Exception as e:
        print(aid, "FAIL", type(e).__name__, e, flush=True)
        time.sleep(5)
        continue
    os.makedirs(dst, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            safe = [m for m in tf.getmembers()
                    if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in m.name]
            tf.extractall(dst, members=safe)
        ok += 1
    except tarfile.ReadError:
        try:
            txt = gzip.decompress(raw)
        except OSError:
            txt = raw
        open(f"{dst}/main.tex", "wb").write(txt)
        ok += 1
    if (n + 1) % 20 == 0:
        print(f"  {n+1}/{len(queue)} ok={ok}", flush=True)
    time.sleep(3)
print("done", ok, "/", len(queue))
