"""Recursive Slop Mitigation over the 10 selected manuscripts, exactly as LAYERS_AND_STEPS_0919.md describes.

Layer 1  editor      Haiku 4.5, frozen
Layer 2  harness     SciSlop.md v0.2 + agent attachment + located instances + round rule + gate + guards
         reviewer    Sonnet 5, a component of the harness, fixed across every manuscript

Resumable. A manuscript whose trajectory.json exists is skipped. Every finished manuscript appends one
section to RESULTS.md and one line to results.jsonl, so an interrupted run continues where it stopped.

  python3 run10.py [--papers FA0002,FA0005] [--workers 4]

Backend. With SH_API_BASE set, every model call goes to that gateway with the key in api.txt. Without it the
calls use the interactive subscription. Nothing else changes.
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent                      # .../slopharness
sys.path.insert(0, str(ROOT / 'temp' / 'code'))
os.environ.setdefault('SH_GATE', '1')
os.environ.setdefault('SH_MODEL', 'claude-haiku-4-5-20251001')
os.environ.setdefault('SH_GATE_MODEL', 'claude-sonnet-5')
os.environ.setdefault('SH_ROUNDS', '3')

import harness as H              # noqa: E402
import quality_gate as Q         # noqa: E402

SKILL = ROOT / 'temp' / 'skill' / 'SciSlop_v0.2_reasonable.md'
RUNS = API / 'runs'
RESULTS_MD = API / 'RESULTS.md'
RESULTS_JSONL = API / 'results.jsonl'
LOG = API / 'run.log'
ARM = 'A-loc'
lock = threading.Lock()


def log(s: str):
    line = time.strftime('%m-%d %H:%M:%S ') + s
    with lock:
        print(line, flush=True)
        with open(LOG, 'a') as f:
            f.write(line + '\n')


def header_if_new():
    if RESULTS_MD.exists():
        return
    sel = json.load(open(API / 'papers10.json'))
    RESULTS_MD.write_text(f"""# Recursive Slop Mitigation. Results of the 10-paper run (append-only file)

One section is appended to this file each time a paper finishes. If interrupted, the next run skips finished papers and continues writing.
Machine-readable form is `results.jsonl`, run trees are `runs/<code>/`, the human-readable comparison is `runs/<code>/CASE_PACK.md`.

## Configuration

| Layer | What | Value |
|---|---|---|
| 1 | Editor (frozen) | {os.environ['SH_MODEL']} |
| 2 | Skill file | {SKILL.name} |
| 2 | Attachment | Agent. As files in the working directory |
| 2 | Location list | 12 per item, candidate in nature |
| 2 | Iteration | 3 rounds, 4 stop conditions |
| 2 | Quality-gate reviewer | {os.environ['SH_GATE_MODEL']} (fixed for all papers) |
| 2 | Retire scope | Sections, manuscript-level evidence gap |
| 0 | Measurers | macro_redund, xsec_ref, citation, evidence_gap (fixed) |

Target {sel['n']} papers, seed {sel['seed']}, stratified sample. Pilot reference {', '.join(sel['pilot_reference'])}.

---
""")


def done_codes() -> set:
    if not RESULTS_JSONL.exists():
        return set()
    out = set()
    for line in open(RESULTS_JSONL):
        line = line.strip()
        if line:
            try:
                out.add(json.loads(line)['code'])
            except Exception:
                pass
    return out


def guard_rollup(t: dict) -> dict:
    hard, soft = {}, {}
    for r in t['rounds']:
        for v in r['guards']['violations']:
            hard[v] = hard.get(v, 0) + 1
        for v in r['guards']['soft_flags']:
            soft[v] = soft.get(v, 0) + 1
    return {'hard': hard, 'soft': soft}


def unused_kinds(code: str) -> dict:
    d = sorted((RUNS / ARM / code / 'measure').glob('R*'), key=lambda p: p.name)
    if not d:
        return {}
    f = d[-1] / 'xsec_ref' / 'objects.jsonl'
    if not f.exists():
        return {}
    out = {}
    for r in (json.loads(l) for l in open(f) if l.strip()):
        if not r.get('reused'):
            out[r['kind']] = out.get(r['kind'], 0) + 1
    return out


def append_result(code: str, t: dict):
    gr = guard_rollup(t)
    ed = round(t['cost_total_usd'], 3)
    gate_cost = round(sum((r['gate']['call'] or {}).get('cost_usd') or 0 for r in t['rounds'] if r.get('gate')), 3)
    kept = sum(r['gate']['kept'] for r in t['rounds'] if r.get('gate'))
    rev = sum(r['gate']['reverted'] for r in t['rounds'] if r.get('gate'))
    ret = sum(len([x for x in (r['gate'].get('retire') or []) if Q.retire_in_scope(x)])
              for r in t['rounds'] if r.get('gate'))
    rec = {'code': code, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'stop': t['stop'], 'rounds': len(t['rounds']),
           'r0': t['r0']['scores'], 'final': t['final_scores'], 'kept': kept, 'reverted': rev, 'retired': ret,
           'guards': gr, 'unused_object_kinds': unused_kinds(code),
           'wall_s': t['wall_total_s'], 'editor_usd': ed, 'reviewer_usd': gate_cost,
           'editor_model': t['model'], 'reviewer_model': os.environ['SH_GATE_MODEL']}
    with lock:
        with open(RESULTS_JSONL, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        md = [f"\n## {code}\n",
              f"Stop {t['stop']}, rounds {len(t['rounds'])}, wall clock {t['wall_total_s']:.0f} s, editing ${ed}, review ${gate_cost}\n",
              '| Item | R0 | Final |', '|---|---|---|']
        for it in H.ITEMS:
            md.append(f"| {it} | {t['r0']['scores'][it]} | {t['final_scores'][it]} |")
        md += ['',
               f"Gate kept {kept}, reverted {rev}, retire applied {ret}.",
               f"Hard guards {gr['hard'] or 'none'}. Soft {gr['soft'] or 'none'}.",
               f"Objects left unused {unused_kinds(code) or 'none'}.", '',
               '| Round | Exec | Kept/Reverted | macro_redund | xsec_ref | citation | evidence_gap |',
               '|---|---|---|---|---|---|---|']
        for r in t['rounds']:
            g = r.get('gate') or {}
            s = r['scores_after']
            md.append(f"| R{r['round']} | {r['exec']} | {g.get('kept','-')}/{g.get('reverted','-')} | "
                      f"{s['macro_redund']} | {s['xsec_ref']} | {s['citation']} | {s['evidence_gap']} |")
        md.append('')
        with open(RESULTS_MD, 'a') as f:
            f.write('\n'.join(md) + '\n')
    return rec


def write_case_pack(code: str):
    """case_pack and placement_audit resolve trees under harness.TEMP, which is the pilot directory. This run
    stores under api/. Point TEMP at api/ while the pack is built, under the lock so threads do not race, and
    restore it. No model call happens here, so holding the lock costs nothing."""
    import io, contextlib, case_pack
    out = RUNS / ARM / code / 'CASE_PACK.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        old_argv, old_temp = sys.argv, H.TEMP
        try:
            H.TEMP = API
            case_pack.H.TEMP = API
            sys.argv = ['case_pack.py', RUNS.name, ARM, code]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                case_pack.main()
            out.write_text(buf.getvalue())
        except Exception as e:
            out.write_text(f'case pack failed: {e!r}')
        finally:
            sys.argv = old_argv
            H.TEMP = old_temp
            case_pack.H.TEMP = old_temp


def run_one(code: str):
    try:
        traj = RUNS / ARM / code / 'trajectory.json'
        if traj.exists():
            t = json.load(open(traj))
        else:
            r0m, m0 = H.r0(code, RUNS)
            log(f'{code} R0 {r0m["scores"]} units { {k: len(v) for k, v in r0m["units"].items()} }')
            t = H.run_paper(ARM, code, SKILL, RUNS, r0m, m0, model=os.environ['SH_MODEL'], log=log)
        write_case_pack(code)
        rec = append_result(code, t)
        log(f'{code} done stop={t["stop"]} final={t["final_scores"]} kept/reverted {rec["kept"]}/{rec["reverted"]} '
            f'hard={rec["guards"]["hard"] or "none"} ${rec["editor_usd"]}+${rec["reviewer_usd"]}')
    except Exception as e:
        import traceback
        log(f'{code} ERROR {e!r}')
        (API / f'error_{code}.txt').write_text(traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', default='')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    header_if_new()
    codes = a.papers.split(',') if a.papers else json.load(open(API / 'papers10.json'))['codes']
    skip = done_codes()
    todo = [c for c in codes if c not in skip]
    log(f'Target {len(codes)} papers, already finished {len(skip)}, to run now {len(todo)}. '
        f'backend={"gateway " + os.environ["SH_API_BASE"] if os.environ.get("SH_API_BASE") else "subscription"}')
    sem = threading.Semaphore(a.workers)

    def worker(c):
        with sem:
            run_one(c)

    ts = [threading.Thread(target=worker, args=(c,)) for c in todo]
    for t in ts:
        t.start(); time.sleep(2)
    for t in ts:
        t.join()
    log(f'Batch finished. Cumulative done {len(done_codes())}/{len(codes)} papers')


if __name__ == '__main__':
    main()
