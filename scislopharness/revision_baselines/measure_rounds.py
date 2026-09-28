"""Measure the four deterministic items over every finished round tree of an arm.

  python3 measure_rounds.py --arm a4_slop --round 1 [--round 3] [--items macro_redund,...] [--force]
Output results/slop/<arm>/R<n>/<item>/{papers.jsonl, <units>.jsonl, summary.json, RUN.json}.
Only rounds whose progress record says exec=ok (or nothing_to_fix, which keeps the tree)
are measured; failed/blocked rounds are left out so they never enter a mean as a value.
Human rows (paired anchors) are included via --hu so summary.json is self-contained.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, ITEMS, ai_codes, load_progress, round_dir, key  # noqa: E402

PY = sys.executable
HERE = Path(__file__).resolve().parent


def measured_codes(arm, n):
    prog = load_progress(arm)
    out = []
    for c in ai_codes():
        r = prog.get(c, {}).get(n)
        if r and r['exec'] in ('ok', 'nothing_to_fix') and round_dir(arm, c, n).exists():
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--round', type=int, action='append', required=True)
    ap.add_argument('--items', default=','.join(ITEMS))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for n in a.round:
        codes = measured_codes(a.arm, n)
        for it in a.items.split(','):
            out = EOR / 'results' / 'slop' / key(a.arm) / f'R{n}' / it
            stamp = out / 'RUN.json'
            if stamp.exists() and not a.force:
                prev = json.load(open(stamp))
                if prev.get('codes') == codes:
                    print(f'{a.arm} R{n} {it}: up to date ({len(codes)})'); continue
            args = [PY, str(HERE / 'measure_tree.py'), '--item', it, '--out', str(out), '--hu']
            for c in codes:
                args += ['--ai', f'{c}={round_dir(a.arm, c, n)}']
            t0 = time.time()
            r = subprocess.run(args, capture_output=True, text=True)
            ok = r.returncode == 0
            print(f'{a.arm} R{n} {it}: {len(codes)} papers {"ok" if ok else "FAIL"} {time.time()-t0:.0f}s', flush=True)
            if not ok:
                (out.parent / f'{it}.err').write_text(r.stdout[-4000:] + r.stderr[-4000:]); continue
            json.dump({'item': it, 'arm': a.arm, 'round': n, 'codes': codes, 'n': len(codes),
                       'ts': time.strftime('%Y-%m-%d %H:%M:%S')}, open(stamp, 'w'), indent=1)


if __name__ == '__main__':
    main()
