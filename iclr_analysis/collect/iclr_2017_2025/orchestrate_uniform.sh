#!/bin/bash
# after texfix: replace tex-less -> wait GPU free -> OCR new -> retire tex-less -> recompute
cd "${SCISLOP_ROOT}/artifact-ai2science/Evaluation/ICLR"
LOG=data/uniform.log; : > $LOG
while screen -ls | grep -q texfix; do sleep 60; done
echo "[$(date +%H:%M)] texfix done; replacing..." >> $LOG
python3 replace_texless.py >> $LOG 2>&1
echo "[$(date +%H:%M)] waiting for ocrall+GPU..." >> $LOG
while screen -ls | grep -q ocrall; do sleep 120; done
while squeue -u $USER -h 2>/dev/null | grep -q .; do sleep 120; done
echo "[$(date +%H:%M)] OCR new papers..." >> $LOG
cd ocr_gpu && WHICH=all sr 4 48 --exclude=hockney --qos=${SLURM_QOS} bash run_ocr.sh >> ../$LOG 2>&1
cd ..
echo "[$(date +%H:%M)] retiring papers w/o tex..." >> $LOG
python3 - >> $LOG 2>&1 <<'PYEOF'
import glob,os,shutil
os.makedirs("data_retired",exist_ok=True); n=0
for mp in glob.glob("data/20*/*/meta.json"):
    d=os.path.dirname(mp)
    if glob.glob(f"{d}/tex/*.tex"): continue
    y=d.split("/")[1]; os.makedirs(f"data_retired/{y}",exist_ok=True)
    shutil.move(d,f"data_retired/{y}/{os.path.basename(d)}"); n+=1
print("retired:",n)
PYEOF
echo "[$(date +%H:%M)] symmetric recompute..." >> $LOG
cd mold_r0/stylo && python3 run_stylo.py >> ../../$LOG 2>&1 \
 && python3 stylo_symmetric.py >> ../../$LOG 2>&1
echo "[$(date +%H:%M)] UNIFORM DATA PIPELINE DONE" >> ../../$LOG
