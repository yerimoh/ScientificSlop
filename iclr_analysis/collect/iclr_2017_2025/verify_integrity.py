"""Gate: every active paper must have tex+pdf+valid ocr.md; exit 1 otherwise."""
import glob,os,sys
bad=[]
for mp in glob.glob("data/20*/*/meta.json"):
    d=os.path.dirname(mp)
    tex=bool(glob.glob(f"{d}/tex/*.tex"))
    pdf=os.path.exists(f"{d}/paper.pdf") and os.path.getsize(f"{d}/paper.pdf")>1000
    ocr=os.path.exists(f"{d}/ocr.md") and os.path.getsize(f"{d}/ocr.md")>500
    if not(tex and pdf and ocr): bad.append((d,tex,pdf,ocr))
print(f"active={len(glob.glob('data/20*/*/meta.json'))} incomplete={len(bad)}")
for b in bad[:10]: print("  BAD",b)
sys.exit(1 if bad else 0)
