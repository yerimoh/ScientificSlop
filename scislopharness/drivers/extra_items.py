"""Adds the two items that were not in the loop: Argument graph and Figure exposition.

Why they are separate.
  argument_graph  needs a labelling call per introduction sentence, and the cache misses when even one sentence changes. Every round
                  needs the Qwen server, and the PMI stage uses a GPU. Without the server the round is recorded as missing.
  fig_exposition  reads only the transcription of the method diagram image. The round tree copies only the tex and the editor has
                  no image tools, so the value is structurally invariant. We therefore reuse the original value instead of remeasuring,
                  but check every round that the figure is still in the manuscript. If it is gone, the round is NA, not 0.

Both items go into the location list given to the editor. fig_exposition cannot be fixed by the editor, so what we observe is
whether the editor leaves it alone or reacts by deleting the figure.
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HERE_CODE = HERE
sys.path.insert(0, str(HERE.parent.parent / 'temp' / 'code'))
import harness as H   # noqa: E402

DRAFT = H.ROOT / 'paper/draft_v6'
BENCH = DRAFT / 'scislopbench/bench165/results/slop'
FIGEXP = DRAFT / 'slop/Artifacts/fig_exposition/results'
EXTRA = ['argument_graph', 'fig_exposition']


# ------------------------------------------------------------------ figure exposition
SOURCES = {'fig_exposition': [FIGEXP / 'papers.jsonl', BENCH / 'fig_exposition' / 'papers.jsonl'],
           'argument_graph': [BENCH / 'argument_graph' / 'papers.jsonl']}


def _bench_row(item: str, code: str) -> dict | None:
    p = next((q for q in SOURCES.get(item, [BENCH / item / 'papers.jsonl']) if q.exists()), None)
    if p is None:
        return None
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('corpus') == 'AI' and r.get('id') == code:
            return r
    return None


def method_figure(code: str) -> str | None:
    """File name of the method diagram image used by the original."""
    row = _bench_row('fig_exposition', code) or {}
    for k in ('figure', 'image', 'file'):
        if row.get(k):
            return str(row[k])
    m0 = H.fars_paper_dir(code)
    for f in H.tex_files(m0):
        mm = re.search(r'\\includegraphics\[[^\]]*\]\{([^}]*framework[^}]*)\}', f.read_text(errors='ignore'))
        if mm:
            return mm.group(1)
    return None


def figure_present(tree: Path, fig: str | None) -> bool:
    if not fig:
        return False
    stem = Path(fig).stem
    return any(stem in f.read_text(errors='ignore') for f in H.tex_files(tree))


FIGEXP_CODE = DRAFT / 'slop/Artifacts/fig_exposition/code'
CENSUS = DRAFT / 'slop/Artifacts/candidte/_census_0914'


def _transcript(code: str) -> list[str]:
    """Transcription of the original diagram. Reads the same cache the measurer uses."""
    import ast
    for p in (CENSUS / 'pairs165_full/transcripts.jsonl', CENSUS / 'pairs165_full/transcripts_fix.jsonl',
              FIGEXP / 'transcripts.jsonl'):
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('key') != f'AI_{code}':
                continue
            r = d.get('result')
            if isinstance(r, str):
                try:
                    r = ast.literal_eval(r)
                except Exception:
                    r = {}
            return (r or {}).get('lines') or []
    return []


def _phrase_norm(t: str) -> str:
    """Normalisation for matching hand-verified phrases against transcription lines. Drops parenthesised notes and symbols."""
    t = re.sub(r'\s*\((?:run configuration|hand-verified)[^)]*\)\s*$', '', str(t or ''))
    return re.sub(r'[^a-z0-9]', '', t.lower())


def _phrase_in(phrase: str, lines: list[str]) -> bool:
    n = _phrase_norm(phrase)
    return bool(n) and any(n in _phrase_norm(l) for l in lines if _phrase_norm(l))


def _read_kinds(lines: list[str], code: str, original_lines: list[str] | None = None):
    """Calls the measurer's verdict function as is. It takes only a list of lines, so it works on transcriptions and diagram sources alike.

    Re-application rule for hand-verified kinds (0919 evening). The measurer adds `experimental_content` from the hand-verified
    table `LEAK.CLEAR[pid]` by pid alone. That verdict concerns a specific phrase in the original image, so on an edited image
    it applies only while the phrase is still present. If `original_lines` is given, contained the phrase, and `lines` does not,
    the kind is dropped. If the phrase cannot be confirmed in the original, the verdict stands. The verdict function itself is unchanged."""
    import importlib.util
    sys.path.insert(0, str(FIGEXP_CODE))
    sys.path.insert(0, str(DRAFT / 'slop/_common'))
    cwd = os.getcwd()
    try:
        os.chdir(FIGEXP_CODE)
        spec = importlib.util.spec_from_file_location('figexp_measure', str(FIGEXP_CODE / 'measure.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        found, words = m.read_kinds(lines, code)
        if original_lines is not None and 'experimental_content' in found:
            phrase = getattr(m, 'LEAK').CLEAR.get(code)
            if phrase and _phrase_in(phrase, original_lines) and not _phrase_in(phrase, lines):
                found = {k: v for k, v in found.items() if k != 'experimental_content'}
        return found, words, m.N_KINDS
    finally:
        os.chdir(cwd)


TIKZ_TEXT = re.compile(r'(?:node|\\node)\s*(?:\[[^\]]*\])?\s*(?:\([^)]*\))?\s*(?:at\s*\([^)]*\))?\s*\{(.+?)\}\s*;?',
                       re.S)
LABELISH = re.compile(r'\\(?:node|draw|path)[^;]*?\{([^{}]{2,200})\}', re.S)


def figure_source(code: str, tree: Path) -> tuple[str, str | None]:
    """What the method diagram is drawn with. ('image', file name), ('tikz', source) or ('missing', None)."""
    fig = method_figure(code)
    stem = Path(fig).stem if fig else 'framework'
    for f in H.tex_files(tree):
        t = f.read_text(errors='ignore')
        for mm in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', t, re.S):
            block = mm.group(1)
            if 'tikzpicture' in block and (stem in block or 'framework' in block.lower()
                                           or re.search(r'\\label\{fig:(framework|overview|method|pipeline|arch)', block)):
                return 'tikz', block
            if stem and stem in block:
                return 'image', fig
    return ('missing', None)


def tikz_lines(src: str) -> list[str]:
    """Extracts only the text the diagram source renders on screen, the same target the measurer reads from a transcription."""
    out = []
    for m in LABELISH.finditer(src):
        t = m.group(1)
        t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
        t = re.sub(r'[{}$]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        if len(t) >= 2:
            out.extend(x.strip() for x in t.split('\\\\') if x.strip())
    return out


def edited_transcript(tree: Path) -> list[str] | None:
    """If the diagram was edited in this round, the post-edit transcription remains in the tree."""
    p = Path(tree) / 'figures' / 'EDIT_REPORT.json'
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text()) or {}).get('lines_after') or None
    except Exception:
        return None


def fig_exposition_score(code: str, tree: Path) -> dict:
    """Three cases. An unedited original image keeps the original value; an image with expository phrases erased is remeasured
    from its transcription; a figure redrawn from source is measured from the source text. In every case the verdict function is the measurer's."""
    row = _bench_row('fig_exposition', code) or {}
    kind, payload = figure_source(code, Path(tree))
    base = {'figure': method_figure(code), 'source_kind': kind}
    if kind == 'missing':
        return {**base, 'slop_score': None, 'figure_present': False, 'kinds': [], 'n_kinds': None,
                'note': 'the method figure is gone from the manuscript'}
    if kind == 'tikz':
        lines = tikz_lines(payload or '')
        found, words, nk = _read_kinds(lines, code)
        return {**base, 'slop_score': round(len(found) / nk, 4), 'figure_present': True,
                'kinds': sorted(found), 'n_kinds': len(found), 'n_lines': len(lines),
                'measured_from': 'figure source', 'note': 'redrawn in an editable source'}
    edited = edited_transcript(Path(tree))
    if edited:
        found, words, nk = _read_kinds(edited, code, original_lines=_transcript(code))
        return {**base, 'slop_score': round(len(found) / nk, 4), 'figure_present': True,
                'kinds': sorted(found), 'n_kinds': len(found), 'n_lines': len(edited),
                'measured_from': 'transcription of the edited image',
                'note': 'the expository text was erased from the original image; every other pixel is the original'}
    return {**base, 'slop_score': row.get('slop_score'), 'figure_present': True,
            'kinds': row.get('kinds') or [], 'n_kinds': row.get('n_kinds'),
            'measured_from': 'cached transcription of the original image',
            'note': 'unedited image, so the value is the original one'}


def fig_exposition_units(code: str, tree: Path) -> list[str]:
    """Gives the phrases to erase verbatim and explains how to erase them.

    Lesson from 0919. Asking for a redraw damaged the mechanism all four times and the gate reverted. Only the expository phrases
    inside the diagram need fixing, so the right move is to keep the original pixels and erase just those spots. The editor only
    picks the phrases; a deterministic procedure does the erasing."""
    kind, payload = figure_source(code, Path(tree))
    if kind == 'missing':
        return []
    edited = edited_transcript(Path(tree))
    lines = edited or (_transcript(code) if kind == 'image' else tikz_lines(payload or ''))
    if not lines:
        return []
    found, _, _ = _read_kinds(lines, code, original_lines=_transcript(code) if edited else None)
    if not found:
        return []
    fig = Path(method_figure(code) or 'the method figure').name
    us = []
    for k in sorted(found):
        for hit in found[k]:
            us.append(f'{fig}: [{k}] this text is expository and has to go. "{str(hit).strip()[:200]}"')
    if kind == 'image':
        us.append(f'{fig}: do not redraw this figure. Write the phrases above, one per line and verbatim, into a '
                  f'file named figures/ERASE.txt in this directory. Everything you list there is erased from the '
                  f'image and every other pixel of it is kept. Write nothing else in that file, and write a phrase '
                  f'only if it appears in the list above.')
    return us


# ------------------------------------------------------------------ argument graph
AG_CACHE = Path(__file__).resolve().parent.parent / 'ag_cache'


def _intro_key(code: str, tree: Path) -> str:
    """AG reads only the introduction. If the introduction is unchanged so is the value, so there is no reason to remeasure."""
    import hashlib
    txt = ''
    for f in H.tex_files(Path(tree)):
        t = f.read_text(errors='ignore')
        if 'introduction' in f.name.lower() or '\\section{Introduction}' in t:
            txt += t
    if not txt:
        txt = H.paper_text(Path(tree))[:8000]
    return hashlib.md5(txt.encode()).hexdigest()[:16]


def argument_graph_measure(code: str, tree: Path, out: Path, stage: str = 'all') -> dict:
    """Runs the measurer code as is on the round tree. Without a server or GPU this is recorded in status.

    Cached by introduction hash. If a round does not touch the introduction the measurement is skipped. This item takes most
    of the round time, so the saving is large."""
    out = Path(out).resolve(); tree = Path(tree).resolve()    # measure_tree chdirs into the checker directory
    out.mkdir(parents=True, exist_ok=True)
    AG_CACHE.mkdir(parents=True, exist_ok=True)
    ck = AG_CACHE / f'{code}_{_intro_key(code, tree)}.json'
    if ck.exists():
        try:
            c = json.loads(ck.read_text())
            import shutil
            for nm in ('papers.jsonl', 'claims.jsonl'):
                src = Path(c.get('dir', '')) / nm
                if src.exists():
                    shutil.copy2(src, out / nm)
            c['rec']['cached'] = True
            return c['rec']
        except Exception:
            pass
    # The PMI stage takes 13 minutes per round on the local CPU. On a GPU it takes a few minutes.
    qos = os.environ.get('SH_AG_QOS', '${SLURM_QOS}')
    ngpu = os.environ.get('SH_AG_GPUS', '4')
    pre = ['sr', ngpu, '48', f'--qos={qos}'] if os.environ.get('SH_AG_SUBMIT', '0') == '1' else []
    r = subprocess.run(pre + [sys.executable, str(H.EOR_CODE / 'measure_tree.py'), '--item', 'argument_graph',
                              '--out', str(out), '--ai', f'{code}={tree}'] +
                       (['--runs', os.environ.get('SH_AG_RUNS', '1')] if os.environ.get('SH_AG_LIGHT', '1') == '1' else []),
                       capture_output=True, text=True, timeout=5400,
                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    if r.returncode != 0:
        (out / 'err.txt').write_text((r.stdout or '')[-4000:] + '\n' + (r.stderr or '')[-4000:])
        return {'slop_score': None, 'status': 'measure_failed'}
    p = out / 'papers.jsonl'
    if not p.exists():
        return {'slop_score': None, 'status': 'no_output'}
    rows = [json.loads(l) for l in open(p) if l.strip()]
    row = next((x for x in rows if x.get('corpus') == 'AI'), rows[0] if rows else {})
    rec = {'slop_score': row.get('slop_score'), 'status': row.get('status'),
           'n_key_claims': row.get('n_key_claims'), 'coverage': row.get('coverage')}
    try:
        ck.write_text(json.dumps({'rec': rec, 'dir': str(out)}, ensure_ascii=False))
    except Exception:
        pass
    return rec


def argument_graph_units(code: str, out: Path) -> list[str]:
    """Declared claims. Rows in claims.jsonl whose status is violation; gives the claim sentence together with the cue sentence that follows it."""
    p = Path(out) / 'claims.jsonl'
    if not p.exists():
        return []
    us = []
    for line in open(p):
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get('status') != 'violation' or (c.get('corpus') and c.get('corpus') != 'AI'):
            continue
        lc = c.get('locus_claim') or {}
        le = c.get('locus_evidence') or {}
        clean = lambda t: re.sub(r'\s+', ' ', str(t or '')).replace('xxcitexx', '[citation]').replace(
            'xxmathxx', '[math]').replace('xxrefxx', '[reference]').strip()
        claim, cue = clean(lc.get('quote')), clean(le.get('quote'))
        if not claim:
            continue
        kind = lc.get('label') or 'key claim'
        us.append(f'[Introduction] the {kind} "{claim[:230]}" is stated before its strongest cue '
                  f'"{cue[:180]}", which currently follows it')
    return us


def server_up() -> bool:
    try:
        import urllib.request
        ep = (H.ROOT / 'artifact-ai2science/_llm/llm_endpoint.txt').read_text().strip()
        urllib.request.urlopen(ep.rstrip('/') + '/models', timeout=8)
        return True
    except Exception:
        return False


def stage_method_figure(code: str, tree: Path, prev: Path | None = None) -> str | None:
    """Copies the single method diagram image into the round tree.

    The round tree copies only the tex, so the editor had never seen the figure. A transcription holds only the text, not which
    box connects where, so a redraw always loses edges. On 0919 FA0002 lost the feedback edge two rounds in a row and the gate
    reverted; that was a design omission, not an editor failure. Other images such as result figures are not staged. They are
    not repair targets and only add context."""
    fig = method_figure(code)
    if not fig:
        return None
    import shutil
    name = Path(fig).name
    # Carry-over (0919 evening). If the previous round tree has an edited diagram, pass it and its edit report on. Copying from
    # the original again would lose the earlier rounds' edits and reset the item every round. ERASE.txt is that round's request,
    # so it is not carried over.
    if prev is not None and (Path(prev) / 'figures' / name).exists():
        dst = Path(tree) / 'figures' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(prev) / 'figures' / name, dst)
        rep = Path(prev) / 'figures' / 'EDIT_REPORT.json'
        if rep.exists():
            shutil.copy2(rep, dst.parent / 'EDIT_REPORT.json')
        return f'figures/{name}'
    src = H.fars_paper_dir(code)
    for cand in (src / fig, src / 'figures' / name):
        if cand.exists():
            dst = Path(tree) / 'figures' / cand.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, dst)
            return f'figures/{cand.name}'
    return None


def apply_erase_request(code: str, tree: Path, log=print) -> dict | None:
    """Executes the figures/ERASE.txt the editor wrote. Phrases not in the list are ignored.

    No generative model is used. It only takes coordinates and fills that rectangle with the surrounding background colour, so no
    pixel is newly drawn and there is no room for fabrication. Afterwards the edited image is re-transcribed with the same prompt and kept in the tree."""
    req = Path(tree) / 'figures' / 'ERASE.txt'
    if not req.exists():
        return None
    fig = method_figure(code)
    img = Path(tree) / 'figures' / Path(fig or '').name
    if not fig or not img.exists():
        return {'status': 'no_image'}
    allowed = set()
    for k, hits in (_read_kinds(_transcript(code), code)[0] or {}).items():
        allowed |= {str(h).strip() for h in hits}
    if rec_prev := edited_transcript(Path(tree)):
        # A phrase already erased has nothing to apply even if requested again. Keep only the remaining phrases in the allow list.
        allowed = {a for a in allowed if _phrase_in(a, rec_prev)}
    asked = [l.strip() for l in req.read_text().splitlines() if l.strip()]
    import re as _re
    strip = lambda t: _re.sub(r'\s*\((?:run configuration|hand-verified)[^)]*\)\s*$', '', t).strip()
    ok, seen = [], set()
    for a2 in asked:
        cand = strip(a2)
        if cand in seen:
            continue
        if any(cand[:50] in strip(b) or strip(b)[:50] in cand for b in allowed):
            ok.append(cand); seen.add(cand)
    
    rejected = [a for a in asked if strip(a) not in ok]
    if not ok:
        return {'status': 'nothing_allowed', 'asked': asked, 'rejected': rejected}
    import subprocess
    out = Path(tree) / 'figures' / img.name
    rep = Path(tree) / 'figures' / 'EDIT_REPORT.json'
    # If this round re-edits an already edited figure, read the previous report for two uses: the start of the derived transcription, and the cumulative history.
    prev_report = {}
    if rep.exists():
        try:
            prev_report = json.loads(rep.read_text()) or {}
        except Exception:
            prev_report = {}
    prior = Path(tree) / 'figures' / '_prior_lines.json'
    prior_lines = prev_report.get('lines_after') or _transcript(code)   # on the first edit, the measurer's original transcription
    if prior_lines:
        prior.write_text(json.dumps(prior_lines, ensure_ascii=False))
    # A GPU is needed only when the reader model is Qwen (SH_FIG_READER=qwen). 32B bf16 needs two 48GB cards. The default reader
    # is the Claude CLI, which finishes on the local CPU.
    qos = os.environ.get('SH_GPU_QOS', '${SLURM_QOS}')
    ngpu = os.environ.get('SH_GPU_GPUS', '2')
    cmd = (['sr', ngpu, '48', f'--qos={qos}'] if os.environ.get('SH_GPU_SUBMIT', '0') == '1' else []) + \
        [sys.executable, str(HERE_CODE / 'figure_edit.py'), code, str(img), str(out), '--report', str(rep)]
    if prior.exists():
        cmd += ['--prior-lines', str(prior)]
    for a in ok:
        cmd += ['--drop', a]
    before = img.read_bytes()
    # Keep a separate copy of the pre-edit image and pass it as input. If input and output are the same file, the outside-target
    # change ratio always reads 0 and the check is meaningless (a trap caught on 0919 evening).
    before_p = Path(tree) / 'figures' / f'_before{img.suffix}'
    before_p.write_bytes(before)
    cmd[cmd.index(str(img))] = str(before_p)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
    if before_p.exists():
        before_p.unlink()
    # Check directly whether the file actually changed. A trap caught on 0919. Even if the subprocess returns success,
    # an unchanged file means no edit happened, and recording that as success would falsify the whole round record.
    after = img.read_bytes() if img.exists() else b''
    changed = after != before
    rec = {'status': ('ok' if (r.returncode == 0 and changed) else
                      ('no_change' if r.returncode == 0 else 'edit_failed')),
           'file_changed': changed, 'asked': asked, 'applied': ok,
           'rejected_not_in_findings': rejected, 'stdout': (r.stdout or '')[-800:],
           'stderr': (r.stderr or '')[-1500:]}
    if rep.exists():
        try:
            new_report = json.loads(rep.read_text()) or {}
            # Cumulative history. Keep what this round erased and what earlier rounds erased in one list. The gate and the
            # post-hoc analysis must be able to see which round erased what.
            if rec['status'] == 'ok':
                hist = list(prev_report.get('history') or [])
                if prev_report and not hist and prev_report.get('erased'):
                    hist.append({'erased': prev_report.get('erased'), 'preservation': prev_report.get('preservation')})
                hist.append({'erased': new_report.get('erased'), 'preservation': new_report.get('preservation'),
                             'requested': asked, 'applied': ok})
                new_report['history'] = hist
                new_report['erased_cumulative'] = [e for h in hist for e in (h.get('erased') or [])]
                rep.write_text(json.dumps(new_report, indent=1, ensure_ascii=False))
            elif prev_report:
                # A failed round does not overwrite the previous report. The derived transcription must keep pointing at the earlier round's state.
                rep.write_text(json.dumps(prev_report, indent=1, ensure_ascii=False))
            rec['report'] = {k: v for k, v in new_report.items() if k not in ('lines_after', 'history')}
        except Exception as e:
            rec['report_error'] = repr(e)
    if prior.exists():
        prior.unlink()
    log(f'[{code}] diagram edit {rec["status"]} requested {len(asked)} applied {len(ok)} file changed {changed}')
    return rec


# ------------------------------------------------------------------ evidence gap. specimen acquisition (T1 quotation)
# 0919 evening. Until now the editor had never seen the record (the round tree holds only tex). The skill's fix says "if the
# record has a specimen, display it", so the record is staged into the editor tree's materials/, and a CLI searcher first finds
# a displayable specimen there and pins it down in SPECIMEN.md. Whether the searcher's quotation is verbatim in that file is
# checked deterministically; if not, it is recorded as not found. If nothing is found the item stays, and that is a result: the record is absent.
SPECIMEN_CACHE = HERE.parent / 'specimen_cache'
SPECIMEN_MODEL = os.environ.get('SH_SPECIMEN_MODEL', 'claude-haiku-4-5-20251001')

# The first Haiku run picked a bug-fix record from development as the "most concrete case". That is not a unit the paper counts.
# A specimen must be one of the units the result tables aggregate; process records (plans, status, bugs, optimisation traces) are excluded.
SPECIMEN_SCOPE = """

한 가지 조건을 더 지켜라. 실물은 원고의 결과 표가 집계하는 단위 가운데 하나여야 한다. 평가에 들어간 입력 하나와
그에 대한 모델 출력, 평가 데이터의 사례 하나, 그 사례에서의 실패 하나 같은 것이다. 개발 과정의 기록은 실물이
아니다. 작업 계획, 실험 상태, 버그 보고와 수정 기록, 최적화 궤적, 요약 보고서, 설정 파일은 아무리 구체적이어도
고르지 마라. 기록에 집계값과 과정 기록만 있고 개별 평가 사례가 없으면 found 를 false 로 적어라. 찾았다면
specimen 안에 "counts_toward": "이 실물이 어느 결과 표의 어느 행/지표에 세어지는가" 를 함께 적어라."""
PROCESS_RECORD = re.compile(r'(^|/)(optimize_trace|traces?|task_plan|FARS_MEMO|\.venv)(/|$)|(^|/)(REPORT|README|task_plan|effectiveness_evaluation_report)\.(md|json)$', re.I)


def _verbatim_in_file(root: Path, rel: str, text: str) -> bool:
    """Is the quotation present verbatim in the named file? Only whitespace is normalised."""
    f = Path(root) / rel
    if not f.is_file():
        return False
    try:
        body = f.read_text(errors='ignore')
    except Exception:
        return False
    norm = lambda t: re.sub(r'\s+', ' ', str(t or '')).strip()
    t = norm(text)
    if not t:
        return False
    if t in norm(body):
        return True
    # In a JSON file escaping can alter characters. Unpack the values and look again.
    try:
        j = json.loads(body)
        flat = []
        def walk(o):
            if isinstance(o, dict):
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
            else:
                flat.append(str(o))
        walk(j)
        return t in norm(' '.join(flat))
    except Exception:
        return False


def ensure_specimen_t2(code: str) -> dict | None:
    """T2. Deterministically picks one instance from the record. No model call, so under a second per paper.

    Called before the T1 search agent. Partly because it is cheap, but also because T1 discards everything under `optimize_trace`
    by path name and so misses the prediction dumps inside it. Returns None when nothing is found and T1 takes over."""
    try:
        sys.path.insert(0, str(HERE))
        import specimen_t2 as T2
        rec = T2.find(code)
    except Exception as e:
        return {'code': code, 'found': False, 'tier': 'T2', 'reason': f'T2 failed {e!r}'}
    if not rec.get('found'):
        return None
    rec['rendered'] = T2.render(rec)
    return rec


def ensure_specimen(code: str, model: str | None = None, log=print) -> dict:
    """Finds one displayable specimen in the project record and caches it. The record does not change between rounds, so once per paper."""
    SPECIMEN_CACHE.mkdir(parents=True, exist_ok=True)
    ck = SPECIMEN_CACHE / f'{code}.json'
    if ck.exists():
        try:
            return json.loads(ck.read_text())
        except Exception:
            pass
    t2 = ensure_specimen_t2(code)
    if t2:
        ck.write_text(json.dumps(t2, ensure_ascii=False, indent=1))
        log(f'[{code}] specimen T2 {t2["grade"]} {t2["file"]} row {t2["row_index"]}/{t2["n_rows"]}')
        return t2
    import time
    sys.path.insert(0, str(HERE))
    import specimen_search as SS
    SS.OUT = HERE.parent / 'specimen_search_loop'     # leave the 0919 03:00 Sonnet search record (specimen_search/) untouched
    SS.OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent.parent / 'temp' / 'code'))
    import quality_gate as Q
    model = model or SPECIMEN_MODEL
    d, meta = SS.build_dir(code)
    res = H.run_cli_json(['-p', SS.INSTRUCTION + SPECIMEN_SCOPE, '--model', model, '--permission-mode', 'acceptEdits',
                          '--add-dir', str(d), '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit',
                          '--settings', str(Q.GATE_SETTINGS), '--restricted'], d, timeout=1500)
    (d / 'searcher_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    rec = {'code': code, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': model, **meta,
           'wall_s': res.get('wall_s'), 'cost_usd': res.get('cost_usd'), 'found': False, 'dir': str(d)}
    f = d / 'SPECIMEN.json'
    if f.exists():
        try:
            j = json.loads(re.sub(r'^```(?:json)?|```$', '', f.read_text().strip(), flags=re.M))
            sp = j.get('specimen') or {}
            rec.update({'found': bool(j.get('found')) and bool(sp), 'reason': j.get('reason'),
                        'searched': j.get('searched'), 'specimen': sp or None})
            if rec['found']:
                rel = str(sp.get('file') or '')
                rel = rel[len('records/'):] if rel.startswith('records/') else rel
                ok = _verbatim_in_file(d / 'records', rel, sp.get('verbatim'))
                rec['verbatim_verified'] = ok
                if ok and PROCESS_RECORD.search(rel):
                    rec['found'] = False
                    rec['reason'] = f'{rel} is a process record (plan, status, bug fix, optimisation trace, report), not an instance the result tables count; treated as not found. ' + str(rec.get('reason') or '')
                elif not ok:
                    rec['found'] = False
                    rec['reason'] = f'searcher quoted text that is not in {rel} verbatim; treated as not found. ' + str(rec.get('reason') or '')
                else:
                    sp['file'] = rel
        except Exception as e:
            rec['parse_error'] = repr(e)
    ck.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    log(f'[{code}] specimen search found={rec["found"]} records {rec.get("n_records")} ${rec.get("cost_usd") or 0:.2f} '
        f'{(rec.get("reason") or "")[:100]}')
    return rec


def stage_editor_materials(code: str, tree: Path, spec: dict | None) -> int:
    """Stages the record into the editor tree's materials/ and pins it down as SPECIMEN.md if a specimen exists, else SPECIMEN_NONE.md."""
    sys.path.insert(0, str(H.EOR_CODE))
    from run_arm import stage_materials
    n = stage_materials(H.fars_paper_dir(code), Path(tree))
    out = Path(tree) / 'materials'
    out.mkdir(exist_ok=True)
    if spec and spec.get('found') and spec.get('tier') == 'T2':
        # T2. One row of the record. Stage the original file as is and pin down which row. Every value is from that file;
        # only the layout is ours. The gate must be able to match it character by character against what is in materials/.
        import shutil
        src = H.fars_paper_dir(code).parent.parent.parent / spec['file']
        rel = f"specimen/{Path(spec['file']).name}"
        if src.is_file():
            (out / 'specimen').mkdir(exist_ok=True)
            shutil.copy2(src, out / 'specimen' / Path(spec['file']).name)
        kind = ('one instance of the evaluation, with its input, the system output and the gold answer'
                if spec.get('grade') == 'instance' else
                'one row of the per-instance record, the unit the result tables are aggregated over')
        (out / 'SPECIMEN.md').write_text('\n'.join([
            '# Specimen taken from the project record', '',
            f'Record file. `materials/{rel}` (project path `{spec["file"]}`).', '',
            f'Which one. Row {spec["row_index"]} of {spec["n_rows"]} at `{spec.get("where") or "the top-level list"}`.', '',
            f'What it is. {kind}.', '',
            'Every value below is copied from that row. Quote the values exactly as they stand, name the record file '
            'beside them, and do not round, shorten or reword a value or add one that is not here.', '',
            'It has to be displayed, not described. A sentence that mentions the values in running prose does not '
            'display them. Put the block inside one of these and nothing else will do. A verbatim, lstlisting, '
            'minted, quote, tcolorbox, mdframed or examplebox environment, or a figure or table whose caption says '
            'in words that it is an example, a case or a failure case.', '',
            '```', str(spec.get('rendered') or ''), '```', '']))
    elif spec and spec.get('found') and spec.get('specimen'):
        sp = spec['specimen']
        src = Path(spec.get('dir', '')) / 'records' / sp['file']
        rel = f"specimen/{Path(sp['file']).name}"
        if src.is_file():
            (out / 'specimen').mkdir(exist_ok=True)
            import shutil
            shutil.copy2(src, out / 'specimen' / Path(sp['file']).name)
        body = [f'# Specimen found in the project record', '',
                f'Record file. `materials/{rel}` (project path `{sp["file"]}`).', '',
                f'What it is. {sp.get("what_it_is") or ""}', '',
                'Verbatim text. Quote exactly this, without changing a character, and name the record file in the caption.', '',
                '```', str(sp.get('verbatim') or ''), '```', '']
        (out / 'SPECIMEN.md').write_text('\n'.join(body))
    else:
        (out / 'SPECIMEN_NONE.md').write_text(
            '# No specimen in the project record\n\nA search of the project\'s own records found no concrete instance '
            '(an input with its output, a case, a failure, a worked example, a quoted specimen) that the manuscript '
            'could display. Reason recorded by the search. ' + str((spec or {}).get('reason') or 'no search result') +
            '\n\nRepair (a) of the Evidence gap entry is not available. Apply repair (b): make the gap explicit where the aggregate '
            'is interpreted, and narrow claims about individual cases to what the aggregate supports. Do not write an example that '
            'was never produced.\n')
    return n


def specimen_note(spec: dict | None) -> str:
    """The one line appended after the evidence gap block of the location list."""
    if spec and spec.get('found'):
        return ('- A displayable specimen exists in the record. See materials/SPECIMEN.md for the verbatim text and its '
                'source file; display it as the skill file says and name the file.\n')
    return ('- No displayable specimen exists in the project record (materials/SPECIMEN_NONE.md), so repair (a) is not '
            'available. Apply repair (b) of the Evidence gap entry: where the aggregate result is interpreted, state that no '
            'individual case was retained or inspected and that the claims rest on aggregate scores only, and narrow sentences '
            'that describe how individual cases behave to what the aggregate supports.\n')
