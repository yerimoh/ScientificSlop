"""Unpack every A4S supplementary.zip that carries LaTeX, one directory per paper.

Nested zips are opened one level down, because several submissions ship the manuscript inside a
second archive. Nothing is judged here; v7_pick_root.py decides which file is the manuscript.
"""
import glob, json, os, sys, zipfile

ROOT = os.environ.get("SCISLOP_ROOT", ".")
A4S = f"{ROOT}/Agents4Science/2025_full"
OUT = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science/a4s_tex"


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for mp in sorted(glob.glob(f"{A4S}/papers/*/*/meta.json")):
        m = json.load(open(mp))
        code = "A4S" + str(m["submission_number"]).zfill(4)
        z = os.path.join(os.path.dirname(mp), "supplementary.zip")
        if not os.path.isfile(z):
            continue
        try:
            zf = zipfile.ZipFile(z)
            names = zf.namelist()
        except Exception:
            continue
        if not any(x.lower().endswith((".tex", ".zip", ".tar.gz", ".tgz")) for x in names):
            continue
        dst = os.path.join(OUT, code)
        os.makedirs(dst, exist_ok=True)
        for x in names:
            if x.startswith("__MACOSX") or x.endswith("/"):
                continue
            try:
                zf.extract(x, dst)
            except Exception:
                pass
        for nz in glob.glob(os.path.join(dst, "**", "*.zip"), recursive=True):
            try:
                zipfile.ZipFile(nz).extractall(os.path.dirname(nz))
            except Exception:
                pass
        if glob.glob(os.path.join(dst, "**", "*.tex"), recursive=True):
            n += 1
        else:
            print(f"  {code}: archive present but no tex")
    print(f"{n} directories with tex -> {OUT}")


if __name__ == "__main__":
    main()
