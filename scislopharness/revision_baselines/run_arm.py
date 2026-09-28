"""Run one revision arm over the benchmark AI papers, R1..R<rounds>, resumable and sharded.

  python3 run_arm.py --arm a1_base|a2_code|a3_review|a4_slop|a4s_<item> [--rounds 3]
                     [--shard i --nshard k] [--codes FA0001,FA0002] [--subset results/subset60.json]
Every round: stage R(n-1) -> R(n), edit, md5 no-op check with one retry, record.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROUNDS, ITEMS, ai_codes, fars_paper_dir, stage, tex_md5, file_md5s, word_count, dangling,
                    run_agent, run_text, load_progress, append_progress, round_dir, runs_dir, write_exec_log,
                    is_session_limit, MODEL, TAG)  # noqa: E402


class SessionLimit(RuntimeError):
    """The editor never ran because the account hit its Claude session limit. The round is
    recorded as blocked (retryable), never as a no-op, and the shard stops so the supervisor
    can wait for the quota window to reset (0830 contamination lesson)."""
import prompts  # noqa: E402

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop'] + [f'a4s_{it}' for it in ITEMS]


# ------------------------------------------------------------------ a1: text in / text out
FILE_RE = re.compile(r'<<<FILE\s+(?P<p>[^>]+?)\s*>>>\n(?P<body>.*?)(?:<<<END\s+(?P=p)\s*>>>|\Z)', re.S)


def a1_files(tree: Path) -> list:
    fs = [tree / 'main.tex']
    if (tree / 'sections').is_dir():
        fs += sorted((tree / 'sections').glob('*.tex'))
    return [f for f in fs if f.exists()]


ONEFILE_RE = re.compile(r'<<<BEGIN FILE>>>\n?(?P<body>.*?)(?:<<<END FILE>>>|\Z)', re.S)


def a1_edit_perfile(tree: Path, n: int, tag: str, logdir: Path) -> dict:
    """One call per file (original 0914 design), used when EOR_A1_MODE=perfile (open-model backend)."""
    files = a1_files(tree)
    payload = [(str(f.relative_to(tree)), f.read_text(errors='ignore')) for f in files]
    rec, dts = {}, 0.0
    for rel, orig in payload:
        out, dt, raw = run_text(prompts.a1_prompt_file(payload, rel, n), prompts.A1_SYSTEM, tree, f'{tag}_{rel.replace("/", "_")}', logdir)
        dts += dt
        if is_session_limit(raw) and os.environ.get('EOR_BACKEND', 'claude') != 'vllm':   # the CLI's quota messages; a paper's own text can contain these words
            return {'limited': True}
        m = ONEFILE_RE.search(out or '')
        new = m.group('body').strip('\n') if m else None
        ok, why = _accept(orig, new, rel)
        if ok:
            (tree / rel).write_text(new); rec[rel] = 'revised'
        else:
            rec[rel] = f'kept_original:{why}'
    return {'files': rec, 'n_revised': sum(1 for v in rec.values() if v == 'revised'), 'mode': 'perfile', 'dt': dts}


def a1_edit(tree: Path, n: int, tag: str, logdir: Path) -> dict:
    if os.environ.get('EOR_A1_MODE', 'single') == 'perfile':
        return a1_edit_perfile(tree, n, tag, logdir)
    files = a1_files(tree)
    payload = [(str(f.relative_to(tree)), f.read_text(errors='ignore')) for f in files]
    out, dt, raw = run_text(prompts.a1_prompt(payload, n), prompts.A1_SYSTEM, tree, tag, logdir)
    if is_session_limit(raw) and os.environ.get('EOR_BACKEND', 'claude') != 'vllm':   # CLI quota messages only; paper text can contain these words
        return {'limited': True}
    blocks = {}
    for m in FILE_RE.finditer(out or ''):
        blocks[m.group('p').strip()] = m.group('body').strip('\n')
    rec = {}
    for rel, orig in payload:
        new = blocks.get(rel)
        ok, why = _accept(orig, new, rel)
        if ok:
            (tree / rel).write_text(new); rec[rel] = 'revised'
        else:
            rec[rel] = f'kept_original:{why}'
    return {'files': rec, 'n_revised': sum(1 for v in rec.values() if v == 'revised'),
            'n_blocks_returned': len(blocks), 'dt': dt}


def _accept(orig: str, new, rel: str):
    if not new:
        return False, 'not_returned'
    ratio = len(new) / max(1, len(orig))
    if ratio < 0.5 or ratio > 2.5:
        return False, f'length_ratio_{ratio:.2f}'
    for tok in set(re.findall(r'\\input\{[^}]*\}|\\begin\{document\}|\\end\{document\}|\\documentclass', orig)):
        if tok not in new:
            return False, f'lost_{tok[:20]}'
    if new.strip() == orig.strip():
        return False, 'identical'
    return True, ''


# ------------------------------------------------------------------ a2 / a3 / a4 agents
def a2_edit(tree, n, tag, logdir):
    ok, dt, raw = run_agent(tree, prompts.A2_PROMPT, tag, logdir, system=None, restricted=True,
                            note='stock system prompt, --restricted')
    return {'cli_ok': ok, 'dt': dt, 'limited': is_session_limit(raw)}


def a3_edit(tree, n, tag, logdir, code):
    from review_qwen import get_review
    review, src = get_review(code, n, tree, logdir)
    if review is None:
        return {'blocked': src}
    ok, dt, raw = run_agent(tree, prompts.a3_prompt(review), tag, logdir, system=prompts.AGENT_SYSTEM)
    return {'cli_ok': ok, 'dt': dt, 'review_source': src, 'review_rating': review.get('rating'),
            'limited': is_session_limit(raw)}


def _apply_blocks(tree: Path, payload: list, out: str) -> dict:
    """Parse <<<FILE p>>> blocks of a text answer and write the accepted ones (a1 guards)."""
    blocks = {}
    for m in FILE_RE.finditer(out or ''):
        blocks[m.group('p').strip()] = m.group('body').strip('\n')
    rec = {}
    for rel, orig in payload:
        ok, why = _accept(orig, blocks.get(rel), rel)
        if ok:
            (tree / rel).write_text(blocks[rel]); rec[rel] = 'revised'
        else:
            rec[rel] = f'kept_original:{why}'
    return {'files': rec, 'n_revised': sum(1 for v in rec.values() if v == 'revised'), 'n_blocks_returned': len(blocks)}


def a4_edit(tree, n, tag, logdir, code, items):
    from slop_units import failing_units
    fb_dir = logdir / f'R{n}_feedback'
    r = failing_units(code, tree, fb_dir, items)
    text_mode = os.environ.get('EOR_BACKEND', 'claude') == 'vllm'      # open-model backend: same issues, a1 text format
    files = a1_files(tree); payload = [(str(f.relative_to(tree)), f.read_text(errors='ignore')) for f in files]
    prompt = prompts.a4_prompt_text(r['units'], items, n, payload) if text_mode else prompts.a4_prompt(r['units'], items, n)
    json.dump({'scores_before': r['scores'], 'n_units': {k: len(v) for k, v in r['units'].items()},
               'prompt': prompt if not text_mode else (prompt.split('===== FILE')[0] if prompt else None), 'mode': 'text' if text_mode else 'agent'},
              open(logdir / f'{tag}_feedback.json', 'w'), indent=1, ensure_ascii=False)
    if prompt is None:
        return {'nothing_to_fix': True, 'scores_before': r['scores']}
    if text_mode:
        out, dt, raw = run_text(prompt, prompts.A1_SYSTEM, tree, tag, logdir)
        info = _apply_blocks(tree, payload, out)
        return {**info, 'mode': 'text', 'dt': dt, 'scores_before': r['scores'], 'n_units': {k: len(v) for k, v in r['units'].items()}}
    ok, dt, raw = run_agent(tree, prompt, tag, logdir, system=prompts.AGENT_SYSTEM)
    return {'cli_ok': ok, 'dt': dt, 'scores_before': r['scores'],
            'n_units': {k: len(v) for k, v in r['units'].items()}, 'limited': is_session_limit(raw)}


def stage_materials(paper_dir: Path, dst: Path, cap: int = 60, cap_bytes: int = 400_000):
    """Copy the project's own experiment records into materials/ for the EOR_EG_V2 ablation.

    Only the records the project produced, never the manuscript sources, and nothing is written
    back to them. Large files are skipped rather than truncated, so nothing the editor reads is a
    partial record it might complete from imagination.
    """
    src = paper_dir.parents[1] / 'exp' / 'EXPERIMENT_RESULTS'
    if not src.is_dir():
        return 0
    out = dst / 'materials'
    out.mkdir(exist_ok=True)
    n = 0
    for f in sorted(src.rglob('*')):
        if not f.is_file() or f.suffix.lower() in ('.tex', '.pdf', '.png', '.jpg', '.pkl', '.pt'):
            continue
        if f.stat().st_size > cap_bytes or n >= cap:
            continue
        rel = f.relative_to(src)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
        n += 1
    return n


# ------------------------------------------------------------------ one paper
def run_paper(arm: str, code: str, rounds: int, shard, done: dict):
    r0 = fars_paper_dir(code)
    base = runs_dir(arm) / code
    logdir = base / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    items = ITEMS if arm == 'a4_slop' else ([arm[4:]] if arm.startswith('a4s_') else None)
    for n in range(1, rounds + 1):
        prev = done.get(code, {}).get(n)
        if prev and prev.get('exec') in ('ok', 'failed', 'nothing_to_fix') and round_dir(arm, code, n).exists():
            continue
        src = r0 if n == 1 else round_dir(arm, code, n - 1)
        if n > 1 and not src.exists():
            append_progress(arm, shard, {'code': code, 'round': n, 'exec': 'blocked', 'reason': 'previous_round_missing',
                                         'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
            return
        dst = stage(src, round_dir(arm, code, n))
        if os.environ.get('EOR_EG_V2') == '1' and items == ['evidence_gap']:
            stage_materials(r0, dst)          # before the md5 snapshot, so it is not a change
        tag = f'{code}_{arm}_R{n}'
        md5_before, files_before = tex_md5(dst), file_md5s(dst)
        w_before, dang_before = word_count(dst), set(dangling(dst))
        t0 = time.time()
        info = {}
        for attempt in range(2):
            if arm == 'a1_base':
                info = a1_edit(dst, n, f'{tag}_a{attempt}', logdir)
            elif arm == 'a2_code':
                info = a2_edit(dst, n, f'{tag}_a{attempt}', logdir)
            elif arm == 'a3_review':
                info = a3_edit(dst, n, f'{tag}_a{attempt}', logdir, code)
            else:
                info = a4_edit(dst, n, f'{tag}_a{attempt}', logdir, code, items)
            if info.get('limited'):
                raise SessionLimit(f'{arm} {code} R{n}')
            if info.get('blocked') or info.get('nothing_to_fix'):
                break
            if tex_md5(dst) != md5_before:
                break
        md5_after, files_after = tex_md5(dst), file_md5s(dst)
        changed = sorted(k for k in files_before if files_after.get(k) != files_before[k]) + \
                  sorted(k for k in files_after if k not in files_before)
        if info.get('blocked'):
            execs = 'blocked'
        elif info.get('nothing_to_fix'):
            execs = 'nothing_to_fix'
        else:
            execs = 'ok' if md5_after != md5_before else 'failed'
        rec = {'code': code, 'round': n, 'arm': arm, 'model': MODEL, 'exec': execs, 'attempts': attempt + 1,
               'dt': round(time.time() - t0, 1), 'md5_before': md5_before, 'md5_after': md5_after,
               'n_files_changed': len(changed), 'changed_files': changed,
               'words_before': w_before, 'words_after': word_count(dst),
               'dangling_new': sorted(set(dangling(dst)) - dang_before), 'info': info,
               'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
        append_progress(arm, shard, rec)
        print(f'[{arm} {code} R{n}] {execs} dt={rec["dt"]:.0f}s files={len(changed)} '
              f'words {w_before}->{rec["words_after"]} dangling+{len(rec["dangling_new"])}', flush=True)
        if execs == 'blocked':
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--codes', default='')
    ap.add_argument('--subset', default='')
    ap.add_argument('--order-by-arms', default='',
                    help='comma list of arms; process papers that those arms already finished first, '
                         'so the set of papers complete in every arm grows as fast as the quota allows')
    a = ap.parse_args()
    codes = ai_codes()
    if a.subset:
        codes = [c for c in codes if c in set(json.load(open(a.subset))['codes'])]
    if a.codes:
        codes = [c for c in codes if c in set(a.codes.split(','))]
    if a.order_by_arms:
        others = [x for x in a.order_by_arms.split(',') if x and x != a.arm]
        have = {c: 0 for c in codes}
        for o in others:
            po = load_progress(o)
            for c in codes:
                if all((po.get(c, {}).get(n) or {}).get('exec') in ('ok', 'nothing_to_fix')
                       for n in range(1, a.rounds + 1)):
                    have[c] += 1
        codes = sorted(codes, key=lambda c: (-have[c], c))
    if a.shard is not None:
        codes = [c for i, c in enumerate(codes) if i % a.nshard == a.shard]
    done = load_progress(a.arm)
    print(f'{a.arm}{TAG}: {len(codes)} papers, rounds={a.rounds}, shard={a.shard}/{a.nshard}, model={MODEL}', flush=True)
    for c in codes:
        try:
            run_paper(a.arm, c, a.rounds, a.shard, done)
        except SessionLimit as ex:
            print(f'{a.arm} shard {a.shard} STOP: session limit at {ex}', flush=True)
            sys.exit(17)
        except Exception as ex:
            append_progress(a.arm, a.shard, {'code': c, 'round': -1, 'exec': 'error', 'reason': repr(ex)[:300],
                                             'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
            print(f'[{a.arm} {c}] ERROR {ex!r}', flush=True)
    print(f'{a.arm} shard {a.shard} DONE', flush=True)


if __name__ == '__main__':
    main()
