"""6-item run. Adds Argument graph and Figure exposition to the 4 deterministic items and measures every round.
The skill is v0.3 (citation definition aligned with the measurer, location cap 40). **The round limit is 3.** It is the value
set by the design document `LAYERS_AND_STEPS_0919.md` and is not changed. Bringing citation down further is solved by the
definition and the location cap, not by adding rounds. Uses the same append rule as run10.py and accumulates results in RESULTS_6ITEMS.md.

  python3 run6items.py --papers FA0002,FA0005 [--workers 2]
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent
sys.path.insert(0, str(ROOT / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
os.environ.setdefault('SH_GATE', '1')
os.environ.setdefault('SH_MODEL', 'claude-haiku-4-5-20251001')
os.environ.setdefault('SH_GATE_MODEL', 'claude-sonnet-5')
os.environ.setdefault('SH_EXTRA_ITEMS', '1')

import harness as H              # noqa: E402
import quality_gate as Q         # noqa: E402
import extra_items as X          # noqa: E402

SKILL = ROOT / 'temp' / 'skill' / Path(os.environ.get('SH_SKILL', 'SciSlop_v0.4.md'))
RUNS = API / os.environ.get('SH_RUNS_DIR', 'runs6')
RESULTS_MD = API / 'RESULTS_6ITEMS.md'
RESULTS_JSONL = API / os.environ.get('SH_RESULTS', 'results_6items.jsonl')
LOG = API / os.environ.get('SH_LOG', 'run6.log')
ARM = 'A-loc'
ALL_ITEMS = H.ITEMS + X.EXTRA
lock = threading.Lock()


def log(s):
    line = time.strftime('%m-%d %H:%M:%S ') + s
    with lock:
        print(line, flush=True)
        with open(LOG, 'a') as f:
            f.write(line + '\n')


def wait_for_quota(max_wait_s: int = 9000):
    """If a session limit is in effect, waits until it lifts. Checked with one cheap call."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        r = H.run_cli_json(['-p', 'ok', '--model', os.environ['SH_MODEL'], '--tools', ''], API, timeout=120)
        if not r.get('limited') and r.get('rc') == 0:
            log('Quota confirmed, starting')
            return True
        log(f'Session limit. Checking again in 300 s ({(time.time()-t0)/60:.0f} min elapsed)')
        time.sleep(300)
    log('Wait time exceeded, aborting')
    return False


def done_codes() -> set:
    if not RESULTS_JSONL.exists():
        return set()
    out = set()
    for line in open(RESULTS_JSONL):
        if line.strip():
            try:
                out.add(json.loads(line)['code'])
            except Exception:
                pass
    return out


def append_result(code: str, t: dict):
    kept = sum(r['gate']['kept'] for r in t['rounds'] if r.get('gate'))
    rev = sum(r['gate']['reverted'] for r in t['rounds'] if r.get('gate'))
    ret = sum(len([x for x in (r['gate'].get('retire') or []) if Q.retire_in_scope(x)])
              for r in t['rounds'] if r.get('gate'))
    hard, soft = {}, {}
    for r in t['rounds']:
        for v in r['guards']['violations']:
            hard[v] = hard.get(v, 0) + 1
        for v in r['guards']['soft_flags']:
            soft[v] = soft.get(v, 0) + 1
    rec = {'code': code, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'stop': t['stop'], 'rounds': len(t['rounds']),
           'per_round': {'R0': t['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in t['rounds']}},
           'extra': {f"R{r['round']}": (r.get('extra') or {}) for r in t['rounds']},
           'kept': kept, 'reverted': rev, 'retired': ret, 'guards': {'hard': hard, 'soft': soft},
           'wall_s': t['wall_total_s'], 'editor_usd': round(t['cost_total_usd'], 3),
           'reviewer_usd': round(sum((r['gate']['call'] or {}).get('cost_usd') or 0
                                     for r in t['rounds'] if r.get('gate')), 3),
           'skill': SKILL.name, 'editor_model': t['model'], 'reviewer_model': os.environ['SH_GATE_MODEL']}
    with lock:
        with open(RESULTS_JSONL, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


def run_one(code: str, rounds: int):
    try:
        traj = RUNS / ARM / code / 'trajectory.json'
        if traj.exists():
            t = json.load(open(traj))
        else:
            r0m, m0 = H.r0(code, RUNS)
            log(f'{code} R0 ' + ', '.join(f'{k} {r0m["scores"].get(k)}' for k in ALL_ITEMS))
            t = H.run_paper(ARM, code, SKILL, RUNS, r0m, m0, model=os.environ['SH_MODEL'],
                            round_limit=rounds, log=log)
        rec = append_result(code, t)
        log(f"{code} done stop={t['stop']} rounds {len(t['rounds'])} kept/reverted {rec['kept']}/{rec['reverted']} "
            f"hard {rec['guards']['hard'] or 'none'} ${rec['editor_usd']}+${rec['reviewer_usd']}")
    except Exception as e:
        import traceback
        log(f'{code} ERROR {e!r}')
        (API / f'error6_{code}.txt').write_text(traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', default='FA0002,FA0005')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--wait-quota', action='store_true')
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    os.environ['SH_ROUNDS'] = str(a.rounds)
    if not X.server_up():
        log('Warning. The label server is down, so argument graph will be recorded as missing')
    if a.wait_quota and not wait_for_quota():
        return
    codes = [c for c in a.papers.split(',') if c and c not in done_codes()]
    log(f'Target {codes}, rounds {a.rounds}, skill {SKILL.name}, items {ALL_ITEMS}')
    ts = [threading.Thread(target=run_one, args=(c, a.rounds)) for c in codes]
    for t in ts:
        t.start(); time.sleep(3)
    for t in ts:
        t.join()
    log('Batch finished')
    import summarize6
    summarize6.main()


if __name__ == '__main__':
    main()
