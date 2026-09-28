#!/bin/bash
# Session-independent pipeline (0915 night): new ICLR papers -> measurers -> views -> detectors (sbatch --wait)
# -> quality filter -> analysis -> reviewer top-up (sbatch --wait) -> figures + report. Log: logs/pipeline_0915.log
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
LOG=$R/logs/pipeline_0915.log; exec >> $LOG 2>&1
step(){ echo; echo "===== [$(date '+%m-%d %H:%M')] $1"; }
step "1 records"; python3 scripts/build_iclr_records.py | tail -3
step "2 measurers (4 items, all papers)"; for c in macro_redund xsec_ref citation evidence_gap; do python3 scripts/run_slop_iclr.py --checker $c 2>&1 | tail -1; done
step "3 prose views"; python3 scripts/build_iclr_texts.py | tail -1
python3 - <<'PY'
import json, glob
R="${SCISLOP_ROOT}/paper/draft_v6/review"
done=set()
for f in glob.glob(f"{R}/results/detectors/binoculars.*.jsonl"):
    done |= {json.loads(l)["id"] for l in open(f)}
n=0
with open(f"{R}/records/texts_iclr_new.jsonl","w") as w:
    for l in open(f"{R}/records/texts_iclr.jsonl"):
        r=json.loads(l)
        if r["id"] not in done and r["body_words"]>=200: w.write(l); n+=1
print("new texts for detectors:", n)
PY
step "4 detectors on new papers (sbatch --wait, 1 GPU, q-mid)"
sbatch --wait --partition=vram48 --qos=${SLURM_QOS} -N 1 -n 1 -c 8 --gres=gpu:1 --job-name=iclr_dnew -o logs/det_new.log --wrap="bash scripts/run_detectors_new.sh $R/records/texts_iclr_new.jsonl new"
echo "detector job exit $?"; grep -c "" results/detectors/binoculars.new.jsonl results/detectors/detectgpt.new.jsonl results/detectors/nts.new.jsonl
step "5 quality audit"; python3 scripts/quality_audit.py
step "6 analysis"; python3 scripts/analyze.py 2>&1 | grep -A 7 "(b) stairs" | head -9
step "7 reviewer top-up subset"; python3 scripts/make_reviewer_topup.py 30
step "8 figures + report (detectors + ours; reviewers as available)"
python3 scripts/fig_stairs.py; python3 scripts/fig_stairs.py post2023; python3 scripts/fig_lines.py
for v in "" "--ours-agg3" "--bands" "--ours-agg3 --bands"; do python3 scripts/fig_rating_systems.py $v | tail -1; done
python3 scripts/fig_stairs_systems.py | tail -1; python3 scripts/report.py
step "9 reviewers top-up (sbatch --wait, two 2-GPU jobs on q-mid)"
J1=$(sbatch --parsable --partition=vram48 --qos=${SLURM_QOS} -N 1 -n 1 -c 16 --gres=gpu:2 --job-name=iclr_b2h_top -o logs/rev_b2h_topup.log --wrap="bash scripts/rev_b2h_topup.sh")
J2=$(sbatch --parsable --partition=vram48 --qos=${SLURM_QOS} -N 1 -n 1 -c 16 --gres=gpu:2 --job-name=iclr_b3a_top -o logs/rev_b3a_topup.log --wrap="bash scripts/rev_b3a_topup.sh")
echo "reviewer jobs $J1 $J2"; while squeue -h -j $J1,$J2 2>/dev/null | grep -q .; do sleep 300; done
echo "reviews now: b2h $(ls results/reviews/b2h | wc -l) b3a $(ls results/reviews/b3a | wc -l)"
step "10 final figures + report"
for v in "" "--ours-agg3" "--bands" "--ours-agg3 --bands"; do python3 scripts/fig_rating_systems.py $v | tail -1; done
python3 scripts/fig_stairs_systems.py | tail -1; python3 scripts/report.py
D="${SCISLOP_ROOT}/paper/draft_v6/paper/real/_ICLR_2027__Scientific_Mold (2)/figures/main_figures"
cp results/fig_review_stairs.pdf "$D/fig3_review_stairs.pdf"; cp results/fig_rating_systems_agg3_bands.pdf "$D/fig3_review_vs_systems.pdf"; cp results/fig_stairs_systems.pdf "$D/fig3_review_stairs_systems.pdf"
step "PIPELINE_DONE"
