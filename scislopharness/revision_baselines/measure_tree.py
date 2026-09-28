"""Run one slop checker (unchanged code) over arbitrary paper trees.
Monkeypatches slop/_common/corpus before the checker is imported (run_slop165.py pattern).

  python3 measure_tree.py --item macro_redund --out DIR --ai FA0001=/path/to/R1 [--ai ...] [--hu]
--hu adds the paired human papers of the given codes (bench165 records) so the checker's
own AI-vs-HU summary is well defined; the HU rows are identical to bench165's.
"""
import argparse, importlib.util, json, os, sys

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f'{ROOT}/paper/draft_v6/slop'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ai_record, hu_record  # noqa: E402

CHECKER_DIR = {
    'macro_redund': f'{SLOP}/Structure/macro_redund/code',
    'xsec_ref': f'{SLOP}/Structure/xsec_ref/code',
    'citation': f'{SLOP}/Argument/citation/code',
    'evidence_gap': f'{SLOP}/Artifacts/evidence_gap/code',
    'argument_graph': f'{SLOP}/Argument/Argument_Graph/code',   # 0916: labels + PMI read from the caches (prefetch_labels.py, pmi_worker.py)
}
EXTRA_ARGV = {'argument_graph': ['--stage', 'pmi']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--item', required=True, choices=list(CHECKER_DIR))
    ap.add_argument('--out', required=True)
    ap.add_argument('--ai', action='append', default=[], help='CODE=DIR')
    ap.add_argument('--hu', action='store_true')
    a, extra = ap.parse_known_args()
    a.extra = extra
    ai_recs, hu_recs = [], []
    for spec in a.ai:
        code, d = spec.split('=', 1)
        from pathlib import Path
        ai_recs.append(ai_record(code, Path(d).resolve()))
        if a.hu:
            h = hu_record(code)
            if h:
                hu_recs.append(h)
    sys.path.insert(0, f'{SLOP}/_common')
    sys.path.insert(0, f'{ROOT}/artifact-ai2science/_llm')
    import corpus  # noqa: E402
    corpus.ai_papers = lambda: ai_recs
    corpus.hu_papers = lambda: hu_recs
    cdir = CHECKER_DIR[a.item]
    sys.path.insert(0, cdir)
    os.chdir(cdir)
    spec = importlib.util.spec_from_file_location(f'measure_{a.item}', os.path.join(cdir, 'measure.py'))
    measure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)
    os.makedirs(a.out, exist_ok=True)
    src = open(os.path.join(cdir, 'measure.py')).read()
    argv = ['measure.py']
    if '"--out"' in src:
        argv += ['--out', a.out]
    else:
        measure.RESULTS = a.out
    argv += EXTRA_ARGV.get(a.item, [])
    argv += a.extra          # extra arguments passed through unchanged to the item measurer (e.g. --runs for argument_graph)
    sys.argv = argv
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        measure.main()
    json.dump({'item': a.item, 'n_ai': len(ai_recs), 'n_hu': len(hu_recs)}, open(os.path.join(a.out, 'RUN.json'), 'w'))


if __name__ == '__main__':
    main()
