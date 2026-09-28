"""Recursive Slop Mitigation, layer 1 (0918 pilot, 3 papers).

Reuses Effects_of_revision/code (staging, md5 no-op, measurers, unit lists, a1 parser) unchanged.
New here: the skill file is the only instruction; two attachment points x two modes; per-call
cost / tokens / turns from the CLI JSON output; guards against M_0 after every round.

  arms   T-def  text editor, definition-only          (SciSlop.md + manuscript -> revised files)
         T-loc  text editor, definition + location    (+ located units in the skill's units format)
         A-def  Claude Code agent, definition-only    (SciSlop.md placed in the manuscript directory)
         A-loc  Claude Code agent, definition + loc.  (+ SLOP_FINDINGS.md in the directory)
"""
from __future__ import annotations
import difflib, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
ROOT = Path(os.environ.get("SCISLOP_ROOT", "."))
EOR_CODE = ROOT / 'paper/draft_v6/Effects_of_revision/code'
sys.path.insert(0, str(EOR_CODE))
from common import (fars_paper_dir, stage, tex_md5, file_md5s, word_count, dangling, cite_keys, bib_keys,   # noqa: E402
                    is_session_limit, clean_env, CLAUDE, ITEMS, tex_files, paper_text)


def backend_env() -> dict:
    """Environment for the CLI subprocess. With SH_API_BASE set, every call goes to that gateway with the key in
    SH_API_KEY_FILE instead of the interactive subscription. Anthropic-compatible gateways need no other change."""
    env = clean_env()
    base = os.environ.get('SH_API_BASE')
    if base:
        env['ANTHROPIC_BASE_URL'] = base
        kf = os.environ.get('SH_API_KEY_FILE', str(TEMP.parent / 'api.txt'))
        key = Path(kf).read_text().strip()
        env['ANTHROPIC_AUTH_TOKEN'] = key
        env['ANTHROPIC_API_KEY'] = key
        env.pop('CLAUDE_CODE_USE_BEDROCK', None)
    return env
import prompts  # noqa: E402
from run_arm import a1_files, FILE_RE, _accept  # noqa: E402
from slop_units import failing_units  # noqa: E402

SEED_SKILL = ROOT / 'paper/draft_v6/Effects_of_revision/skill/SciSlop.md'
EDITOR = os.environ.get('SH_MODEL', 'claude-haiku-4-5-20251001')
ROUND_LIMIT = int(os.environ.get('SH_ROUNDS', '3'))

# skill entry heading -> measurer key (entries 3 and 5 have no deterministic measurer in this pilot)
SKILL_ITEM_OF = {'Cross-section references': 'xsec_ref', 'Macro redundancy': 'macro_redund',
                 'Argument graph': 'argument_graph', 'Citation': 'citation',
                 'Figure exposition': 'fig_exposition', 'Evidence gap': 'evidence_gap'}
ITEM_HEADING = {v: k for k, v in SKILL_ITEM_OF.items()}


# ------------------------------------------------------------------ skill file
def parse_skill(path: Path) -> dict:
    """Frontmatter fields, global rules, and the four fields of every entry (for the validator and the report)."""
    txt = Path(path).read_text()
    fm = {}
    m = re.match(r'---\n(.*?)\n---\n', txt, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ':' in line:
                k, v = line.split(':', 1); fm[k.strip()] = v.strip()
    entries = {}
    for em in re.finditer(r'^### \d+\. (?P<head>[^(\n]+?)\s*(?:\((?P<paren>[^)]*)\))?\s*\n(?P<body>.*?)(?=^### |\Z|^---)', txt, re.S | re.M):
        head = em.group('head').strip(); body = em.group('body')
        f = {}
        for field in ('Definition', 'Fix', 'Units as reported'):
            fm2 = re.search(r'\*\*' + field + r'\.\*\*\s*(.*?)(?=\n\s*\n\*\*|\Z)', body, re.S)
            f[field.lower().split()[0]] = re.sub(r'\s+', ' ', fm2.group(1)).strip() if fm2 else ''
        entries[head] = {'paren': em.group('paren'), **f}
    rules = re.search(r'## Global rules\n(.*?)\n---', txt, re.S)
    return {'frontmatter': fm, 'entries': entries, 'global_rules': rules.group(1).strip() if rules else '',
            'n_chars': len(txt), 'md5': hashlib.md5(txt.encode()).hexdigest()}


def location_cap(skill_path: Path) -> int:
    if os.environ.get('SH_LOCATION_CAP', '').isdigit():      # override for the per-round time cap. Recorded as location_cap
        return int(os.environ['SH_LOCATION_CAP'])
    fm = parse_skill(skill_path)['frontmatter']
    try:
        return int(fm.get('location_cap', '12'))
    except ValueError:
        return 12


def location_mode(skill_path: Path) -> str:
    v = parse_skill(skill_path)['frontmatter'].get('location_mode', 'orders').strip().lower()
    return 'candidates' if v == 'candidates' else 'orders'


def calls_per_round(skill_path: Path) -> str:
    """'joint' (default): one editor turn per round with every entry's instances.
    'per_entry': one editor turn per entry that has instances, applied in the skill file's order (layer-2 policy knob)."""
    v = parse_skill(skill_path)['frontmatter'].get('calls_per_round', 'joint').strip().lower()
    return 'per_entry' if v == 'per_entry' else 'joint'


# ------------------------------------------------------------------ located units in the skill's format
def findings_text(units: dict, cap: int, r: int, only: str | None = None, mode: str = 'orders') -> str | None:
    """Located instances, one block per skill entry, in the order of the skill file. None if nothing fails.
    mode 'orders' (v0.1 header) or 'candidates' (skill frontmatter location_mode: candidates, v0.2)."""
    order = ['xsec_ref', 'macro_redund', 'argument_graph', 'citation', 'fig_exposition', 'evidence_gap']
    if only:
        order = [only]
    if mode == 'candidates':
        parts = [f'Located instances (external check, round {r}). These are the places where the check still finds the pattern. '
                 f'Consider each one and repair it where the repair reads naturally, as the skill file says; leave an instance unchanged '
                 f'when the only available repair would add text the argument does not need. Instances are given in the format of the '
                 f'entry\'s "Units as reported" field.\n']
    else:
        parts = [f'Located instances (external check, round {r}). Only these instances remain; treat every listed '
                 f'instance as the skill file says. Instances are given in the format of the entry\'s "Units as reported" field.\n']
    k = 0
    for it in order:
        us = units.get(it) or []
        if not us:
            continue
        k += 1
        parts.append(f'\n## {ITEM_HEADING[it]}\n')
        if it == 'evidence_gap':
            parts.append('- Present. ' + us[0] + '\n')
        else:
            parts.append(f'{len(us)} instances.\n')
            for u in us[:cap]:
                parts.append(f'- {u}\n')
            if len(us) > cap:
                parts.append(f'- (and {len(us) - cap} more of the same kind; treat them the same way)\n')
    return ''.join(parts) if k else None


# ------------------------------------------------------------------ CLI with JSON usage
def run_cli_json(args: list, cwd: Path, timeout: int = 1800, env_extra: dict | None = None) -> dict:
    t0 = time.time()
    try:
        r = subprocess.run([str(CLAUDE)] + args + ['--output-format', 'json'], cwd=str(cwd), env={**backend_env(), **(env_extra or {})},
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, so, se = -1, '', 'TIMEOUT'
    dt = time.time() - t0
    out = {'rc': rc, 'wall_s': round(dt, 1), 'stderr': (se or '')[:3000], 'result': None, 'usage': None}
    try:
        j = json.loads(so)
        out.update({'result': j.get('result'), 'cost_usd': j.get('total_cost_usd'), 'duration_ms': j.get('duration_ms'),
                    'duration_api_ms': j.get('duration_api_ms'), 'num_turns': j.get('num_turns'),
                    'usage': j.get('usage'), 'model_usage': j.get('modelUsage'), 'is_error': j.get('is_error'),
                    'stop_reason': j.get('stop_reason'), 'subtype': j.get('subtype')})
    except Exception:
        out['result'] = so
        out['json_parse_failed'] = True
    out['limited'] = bool(is_session_limit((out.get('result') or '') + (se or '') if rc != 0 or out.get('is_error') else (se or '')))
    return out


def think_env(env_name: str) -> dict:
    """Thinking-token cap for model calls. Measured on 0919 evening: one Haiku edit call spent 110 s on 15k thinking tokens,
    and with the cap at 0 it returns the same number of edits in 13 s. The CLI reads the MAX_THINKING_TOKENS environment variable."""
    v = os.environ.get(env_name, '')
    return {'MAX_THINKING_TOKENS': v} if v.isdigit() else {}


def run_text_json(prompt: str, system: str, cwd: Path, model: str = EDITOR) -> dict:
    return run_cli_json(['-p', prompt, '--model', model, '--tools', '', '--system-prompt', system], cwd, timeout=1500,
                        env_extra=think_env('SH_EDITOR_THINK'))


def run_agent_json(prompt: str, cwd: Path, model: str = EDITOR) -> dict:
    # stock Claude Code system prompt, --restricted (no shell), no web, edits auto-accepted: same wrapper as arm a2_code
    mt = os.environ.get('SH_EDITOR_MAX_TURNS', '')
    ef = os.environ.get('SH_EDITOR_EFFORT', '').strip().lower()
    return run_cli_json(['-p', prompt, '--model', model, '--permission-mode', 'acceptEdits', '--add-dir', str(cwd),
                         '--disallowed-tools', 'WebSearch,WebFetch', '--restricted']
                        + (['--max-turns', mt] if mt.isdigit() and int(mt) > 0 else [])
                        + (['--effort', ef] if ef in ('low', 'medium', 'high') else []), cwd, timeout=1800,
                        env_extra=think_env('SH_EDITOR_THINK'))


def run_agents_parallel(prompt: str, dst: Path, model: str, code: str, rec: dict) -> dict:
    """Call the editor per file, in parallel (0919 evening, 5-minute cap per round).

    One agent fixing the whole manuscript in sequence chains as many reads as files plus one turn per edit, which takes minutes.
    Instead we make one copy of the manuscript per file, and in each copy one agent reads the same SciSlop.md and SLOP_FINDINGS.md
    but fixes **only its own file**. Afterwards file k is taken from copy k and merged. The copies are separate, so there are no
    conflicts, and wall clock becomes the single slowest file. The diagram's ERASE.txt is written by the agent that owns the file
    carrying the diagram. Edits to other files are discarded, and the instruction says so."""
    import threading
    files = [f for f in a1_files(dst)]
    if not files:
        return run_agent_json(prompt, dst, model)
    # The file carrying the method diagram. Only that agent may write figures/ERASE.txt. The file is located by the
    # diagram's file name. Picking the first file with \includegraphics would catch a section with some other figure, and
    # then the editor that actually owns the diagram is told not to write, so the round never touches it (this lost R1 and R2 on 0919 evening).
    fig_file = None
    try:
        sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
        import extra_items as _XF
        stem = Path(_XF.method_figure(code) or '').stem
    except Exception:
        stem = ''
    if stem:
        fig_file = next((f for f in files if stem in f.read_text(errors='ignore')), None)
    if fig_file is None:
        fig_file = next((f for f in files if '\\includegraphics' in f.read_text(errors='ignore')), None)
    snap = {f: f.read_bytes() for f in files}
    work = dst.parent / f'_par_{dst.name}'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    results = {}

    def one(f: Path):
        rel = str(f.relative_to(dst))
        cp = work / rel.replace('/', '__')
        shutil.copytree(dst, cp, ignore=shutil.ignore_patterns('_bands', '_par_*'))
        p = (prompt + f' You are one of several editors working in parallel on this manuscript, and your file is {rel}. '
             f'Read the other files for context but edit only {rel}; edits to any other file are discarded. Repair the listed '
             f'instances whose natural home is in {rel}. Do not create new files, except that ')
        if fig_file is not None and f == fig_file:
            p += 'you alone may write figures/ERASE.txt as the Figure exposition instructions say.'
        else:
            p += 'you must not write figures/ERASE.txt; another editor does that.'
        r = run_agent_json(p, cp, model)
        r['_prompt'] = p
        results[rel] = (cp, r)

    ts = [threading.Thread(target=one, args=(f,)) for f in files]
    for t in ts:
        t.start(); time.sleep(1.0)
    for t in ts:
        t.join()
    # Merge. Only file k from copy k, and only ERASE.txt from the diagram file's copy
    per_file = {}
    for rel, (cp, r) in results.items():
        src = cp / rel
        if src.exists() and src.read_bytes() != snap[dst / rel]:
            (dst / rel).write_bytes(src.read_bytes()); per_file[rel] = 'revised'
        else:
            per_file[rel] = 'unchanged'
        if fig_file is not None and rel == str(fig_file.relative_to(dst)) and (cp / 'figures' / 'ERASE.txt').exists():
            (dst / 'figures').mkdir(exist_ok=True)
            shutil.copy2(cp / 'figures' / 'ERASE.txt', dst / 'figures' / 'ERASE.txt'); per_file['figures/ERASE.txt'] = 'written'
    rec['_per_file_io'] = {rel: (r.get('_prompt', ''), r.get('result') or '') for rel, (cp, r) in results.items()}
    rec['parallel_editor'] = {'files': per_file,
                              'calls': {rel: {k: r.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'num_turns', 'is_error', 'limited')}
                                        for rel, (cp, r) in results.items()}}
    shutil.rmtree(work, ignore_errors=True)
    rs = [r for cp, r in results.values()]
    usage = {}
    for r in rs:
        for kk, v in (r.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[kk] = usage.get(kk, 0) + v
    return {'rc': max((r.get('rc') or 0) for r in rs), 'wall_s': round(max((r.get('wall_s') or 0) for r in rs), 1),
            'cost_usd': sum((r.get('cost_usd') or 0) for r in rs), 'num_turns': sum((r.get('num_turns') or 0) for r in rs),
            'duration_ms': max((r.get('duration_ms') or 0) for r in rs), 'usage': usage,
            'is_error': any(r.get('is_error') for r in rs), 'limited': any(r.get('limited') for r in rs),
            'result': '\n\n'.join(f'##### {rel}\n' + str(r.get('result') or '') for rel, (cp, r) in results.items()),
            'stderr': '\n'.join(str(r.get('stderr') or '') for cp, r in results.values())[:3000]}


def _figure_file(dst: Path, files: list) -> Path | None:
    for f in files:
        t = f.read_text(errors='ignore')
        if '\\includegraphics' in t and 'figure' in t:
            return f
    return None


def run_text_parallel(skill_txt: str, findings: str | None, notes: str | None, r: int, dst: Path, model: str,
                      code: str, rec: dict) -> dict:
    """Fastest editor attachment. Send one text call per file in parallel and receive the whole fixed file back.

    (0919 evening, 5-minute cap per round.) The agent attachment takes 3 minutes because reading and fixing files spans dozens
    of turns. The text attachment packs the skill file, the location list and that one file into a single message and finishes
    in one turn. Files are called separately and concurrently, so wall clock is the single longest file. The skill file and the
    location list are also left in the tree for the record. Only the call for the file carrying the diagram may return a
    figures/ERASE.txt block, and a later deterministic step checks its content against the location list and erases only permitted phrases."""
    import threading
    files = a1_files(dst)
    fig_file = _figure_file(dst, files)
    extra_notes = ('\n' + notes) if notes else ''
    results = {}

    def one(f: Path):
        rel = str(f.relative_to(dst))
        txt = f.read_text(errors='ignore')
        prompt = text_prompt(skill_txt, [(rel, txt)], r, (findings or '') + extra_notes)
        prompt += (f'\nOnly this one file, {rel}, is given to you; other editors handle the other files in parallel. Repair the '
                   f'located instances whose natural home is in {rel} and leave the rest to the others. When nothing in the list '
                   f'belongs in this file, return the file unchanged. ')
        if fig_file is not None and f == fig_file:
            prompt += ('The method figure is a raster image and cannot be redrawn here. For the Figure exposition entry, return a '
                       'second block, <<<FILE figures/ERASE.txt>>> ... <<<END figures/ERASE.txt>>>, whose content is the '
                       'expository phrases to erase, one per line, copied verbatim from the located instances; a deterministic '
                       'step erases exactly those phrases from the image and keeps every other pixel. Omit the block when there '
                       'is nothing to erase. ')
        else:
            prompt += 'Do not return a figures/ERASE.txt block; another editor does that. '
        res = run_text_json(prompt, TEXT_SYSTEM, dst, model)
        res['_prompt'] = prompt
        results[rel] = (txt, res)

    ts = [threading.Thread(target=one, args=(f,)) for f in files]
    for t in ts:
        t.start(); time.sleep(0.5)
    for t in ts:
        t.join()
    acc = {}
    for rel, (orig, res) in results.items():
        blocks, _parser = parse_blocks(res.get('result') or '')
        ok, why = _accept(orig, blocks.get(rel), rel)
        if ok:
            (dst / rel).write_text(blocks[rel]); acc[rel] = 'revised'
        else:
            acc[rel] = f'kept_original:{why}'
        if fig_file is not None and rel == str(fig_file.relative_to(dst)) and blocks.get('figures/ERASE.txt', '').strip():
            (dst / 'figures').mkdir(exist_ok=True)
            (dst / 'figures' / 'ERASE.txt').write_text(blocks['figures/ERASE.txt'].strip() + '\n'); acc['figures/ERASE.txt'] = 'written'
    rec['policy'] = 'text_parallel'; rec['files'] = acc
    rec['_per_file_io'] = {rel: (res.get('_prompt', ''), res.get('result') or '') for rel, (o, res) in results.items()}
    rec['parallel_editor'] = {'calls': {rel: {k: res.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'num_turns', 'is_error', 'limited')}
                                        for rel, (o, res) in results.items()}}
    rs = [res for o, res in results.values()]
    usage = {}
    for x in rs:
        for kk, v in (x.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[kk] = usage.get(kk, 0) + v
    return {'rc': max((x.get('rc') or 0) for x in rs) if rs else -1, 'wall_s': round(max((x.get('wall_s') or 0) for x in rs), 1) if rs else 0,
            'cost_usd': sum((x.get('cost_usd') or 0) for x in rs), 'num_turns': sum((x.get('num_turns') or 0) for x in rs),
            'duration_ms': max((x.get('duration_ms') or 0) for x in rs) if rs else 0, 'usage': usage,
            'is_error': any(x.get('is_error') for x in rs), 'limited': any(x.get('limited') for x in rs),
            'result': '\n\n'.join(f'##### {rel}\n' + str(x.get('result') or '') for rel, (o, x) in results.items()),
            'stderr': '\n'.join(str(x.get('stderr') or '') for o, x in results.values())[:3000]}


EDIT_BLOCK_RE = re.compile(r'<<<EDIT>>>\s*\n<<<OLD>>>\n(?P<old>.*?)\n<<<NEW>>>\n(?P<new>.*?)\n<<<END>>>', re.S)
ERASE_BLOCK_RE = re.compile(r'<<<ERASE>>>\n(?P<body>.*?)\n<<<END>>>', re.S)

EDITS_FORMAT = """Return only the edits, using exactly this format and nothing else. Each edit replaces one contiguous passage
of the file with new text; OLD must be copied character for character from the file (a whole sentence or paragraph, long
enough to occur exactly once), and NEW is what replaces it (empty NEW deletes the passage):

<<<EDIT>>>
<<<OLD>>>
(exact original passage)
<<<NEW>>>
(revised passage)
<<<END>>>

Return as many edit blocks as needed and no other text. If nothing in this file needs to change, return the single word
NOCHANGE."""


def apply_edits(orig: str, out: str) -> tuple[str, dict]:
    """Apply edit blocks deterministically. Replace only when OLD occurs exactly once. Otherwise skip and record the reason."""
    txt = orig
    log = {'applied': 0, 'skipped': []}
    for m in EDIT_BLOCK_RE.finditer(out or ''):
        old, new = m.group('old'), m.group('new')
        if not old.strip():
            log['skipped'].append('empty_old'); continue
        n = txt.count(old)
        if n == 1:
            txt = txt.replace(old, new, 1); log['applied'] += 1
        else:
            # Retry once for the case where only whitespace differs
            norm = re.sub(r'\s+', ' ', old).strip()
            cands = [mm for mm in re.finditer(re.escape(norm).replace(r'\ ', r'\s+'), txt)]
            if len(cands) == 1:
                a, b = cands[0].span(); txt = txt[:a] + new + txt[b:]; log['applied'] += 1
            else:
                log['skipped'].append(f'old_found_{n}_times:{old[:60]!r}')
    return txt, log


def run_edits_parallel(skill_txt: str, findings: str | None, notes: str | None, r: int, dst: Path, model: str,
                      code: str, rec: dict, units: dict | None = None) -> dict:
    """Fastest editor attachment. Send one text call per file in parallel, but receive **only edit blocks** rather than the
    whole fixed file (0919 evening). Returning whole files costs output tokens proportional to file length and took minutes on long sections.
    Edit blocks carry only the changed sentences, so output is short and application is deterministic (only when OLD occurs exactly once).
    It gathers in one turn what the agent's Edit tool would do. The diagram file's call may return phrases to erase in a <<<ERASE>>> block."""
    import threading
    files = a1_files(dst)
    fig_file = _figure_file(dst, files)
    extra_notes = ('\n' + notes) if notes else ''
    results = {}
    # Per-file repair targets (0920). Cross-section reference, isolated citation and argument order barely moved because the
    # editor did not know which sentence in its file to fix. Attach candidate sentences found deterministically in the manuscript, per file. The editor judges, the gate decides what is kept.
    tg = {}
    if os.environ.get('SH_TARGETS', '1') == '1' and units:
        try:
            sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
            import targets as _T
            tg = _T.build(dst, units)
            cap = int(os.environ.get('SH_TARGETS_CAP', '8'))
            tg = {k: v[:cap] for k, v in tg.items()}
            rec['targets'] = {k: len(v) for k, v in tg.items()}
        except Exception as e:
            rec['targets_error'] = repr(e)

    def one(f: Path):
        rel = str(f.relative_to(dst))
        txt = f.read_text(errors='ignore')
        others = ''
        if os.environ.get('SH_EDITOR_CONTEXT', '1') == '1':
            # The other files are given as read-only context. Knowing which section uses which object is needed to write pointer sentences.
            others = '\n'.join(f'===== CONTEXT {str(g.relative_to(dst))} (read-only) =====\n{g.read_text(errors="ignore")[:40000]}'
                                for g in files if g != f)
            others = f'\nThe other files of the manuscript follow for context only; do not return edits for them.\n{others}\n\n'
        if os.environ.get('SH_EDITOR_GENERIC') == '1':
            # Gate-only arm (0922 paragraph-3 ablation, E7). No skill file, no location list; only the same generic improvement instruction as the Claude Code comparison condition.
            # The reviewer still receives the full skill and location list (SH_GATE_SKILL, SH_GATE_FULL_FINDINGS).
            head = (f'{prompts.GENERIC} This is revision round {r}. One LaTeX source file of the manuscript follows. Only this one file, '
                    f'{rel}, is yours; other editors handle the other files in parallel, so edit only what belongs in {rel}.\n\n'
                    + (f'===== REVIEW_NOTES.md =====\n{notes}\n===== END REVIEW_NOTES.md =====\n\n' if notes else '') + others)
        elif findings:
            head = (f'A skill file, SciSlop.md, the located instances, and one LaTeX source file of a manuscript follow. Read the skill '
                    f'file, then revise this file as the skill file says. This is revision round {r}. Only this one file, {rel}, is '
                    f'yours; other editors handle the other files in parallel. Repair the located instances whose natural home is in '
                    f'{rel} and leave the rest to the others.\n\n'
                    f'===== SciSlop.md =====\n{skill_txt}\n===== END SciSlop.md =====\n\n'
                    f'===== SLOP_FINDINGS.md =====\n{(findings or "")}{extra_notes}\n===== END SLOP_FINDINGS.md =====\n\n' + others)
        else:
            # Attachment without locations (0922 component ablation, SH_EDITS_PARALLEL_DEF=1). Only the skill file and the one file are given; the editor
            # finds the instances itself following the skill file. Reviewer notes (reverted changes) are not a location list, so they are attached if present.
            head = (f'A skill file, SciSlop.md, and one LaTeX source file of a manuscript follow. Read the skill file, then revise this '
                    f'file as the skill file says. This is revision round {r}. No located instances are given; find the instances in '
                    f'this file yourself as the skill file says. Only this one file, {rel}, is yours; other editors handle the other '
                    f'files in parallel. Repair the instances whose natural home is in {rel} and leave the rest to the others.\n\n'
                    f'===== SciSlop.md =====\n{skill_txt}\n===== END SciSlop.md =====\n\n'
                    + (f'===== REVIEW_NOTES.md =====\n{notes}\n===== END REVIEW_NOTES.md =====\n\n' if notes else '') + others)
        fig_note = ''
        if fig_file is not None and f == fig_file and findings and '## Figure exposition' in findings:
            fig_note = ('\nThe method figure is included from this file. It is a raster image and cannot be redrawn, so the Figure '
                        'exposition entry is repaired differently: in addition to any edit blocks, return one block\n<<<ERASE>>>\n'
                        '(the expository phrases listed under "Figure exposition" in the located instances, one per line, copied '
                        'verbatim)\n<<<END>>>\nA deterministic step erases exactly those phrases from the image and keeps every other '
                        'pixel. Leave a phrase out only if you judge it to be a component of the mechanism rather than exposition. '
                        'Return this block even when the file itself needs no edit.\n')
        tblock = ''
        if tg.get(rel):
            tblock = ('\n===== TARGETS IN THIS FILE =====\nEach line names a sentence of this file where an external check found a natural '
                      'place for a repair. Decide each one: repair it where it reads naturally, and leave it when the repair would say '
                      'something the manuscript does not already say. Prefer repairing over leaving; a reviewer reverts anything unsupported.\n'
                      + '\n'.join('- ' + x for x in tg[rel]) + '\n===== END TARGETS =====\n')
        prompt = head + tblock + EDITS_FORMAT + fig_note + f'\n===== FILE {rel} =====\n{txt}\n===== END FILE =====\n'
        res = run_text_json(prompt, TEXT_SYSTEM, dst, model)
        res['_prompt'] = prompt
        results[rel] = (txt, res)

    ts = [threading.Thread(target=one, args=(f,)) for f in files]
    for t in ts:
        t.start(); time.sleep(0.3)
    for t in ts:
        t.join()
    acc = {}; edit_log = {}
    for rel, (orig, res) in results.items():
        out = res.get('result') or ''
        new, lg = apply_edits(orig, out)
        edit_log[rel] = lg
        if lg['applied'] and new != orig:
            ok, why = _accept(orig, new, rel)
            if ok:
                (dst / rel).write_text(new); acc[rel] = f"revised:{lg['applied']}"
            else:
                acc[rel] = f'kept_original:{why}'
        else:
            acc[rel] = 'unchanged' if 'NOCHANGE' in out or not lg['applied'] else 'unchanged'
        if not (dst / 'figures' / 'ERASE.txt').exists():
            m = ERASE_BLOCK_RE.search(out)
            if m and m.group('body').strip():
                (dst / 'figures').mkdir(exist_ok=True)
                (dst / 'figures' / 'ERASE.txt').write_text(m.group('body').strip() + '\n'); acc['figures/ERASE.txt'] = 'written'
    rec['policy'] = 'edits_parallel'; rec['files'] = acc; rec['edit_log'] = edit_log
    rec['_per_file_io'] = {rel: (res.get('_prompt', ''), res.get('result') or '') for rel, (o, res) in results.items()}
    rec['parallel_editor'] = {'calls': {rel: {k: res.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'num_turns', 'is_error', 'limited')}
                                        for rel, (o, res) in results.items()}}
    rs = [res for o, res in results.values()]
    usage = {}
    for x in rs:
        for kk, v in (x.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[kk] = usage.get(kk, 0) + v
    return {'rc': max((x.get('rc') or 0) for x in rs) if rs else -1, 'wall_s': round(max((x.get('wall_s') or 0) for x in rs), 1) if rs else 0,
            'cost_usd': sum((x.get('cost_usd') or 0) for x in rs), 'num_turns': sum((x.get('num_turns') or 0) for x in rs),
            'duration_ms': max((x.get('duration_ms') or 0) for x in rs) if rs else 0, 'usage': usage,
            'is_error': any(x.get('is_error') for x in rs), 'limited': any(x.get('limited') for x in rs),
            'result': '\n\n'.join(f'##### {rel}\n' + str(x.get('result') or '') for rel, (o, x) in results.items()),
            'stderr': '\n'.join(str(x.get('stderr') or '') for o, x in results.values())[:3000]}


# ------------------------------------------------------------------ instructions (no score, no threshold; the file is the instruction)
LENIENT_RE = re.compile(r'^=+\s*FILE\s+(?P<p>\S+?)\s*=+\s*\n(?P<body>.*?)(?=^=+\s*(?:FILE\s+\S+|END(?:\s+\S+)?)\s*=+\s*$|\Z)', re.S | re.M)


START_RE = re.compile(r'^(?:<<<FILE\s+(?P<a>[^>]+?)\s*>>>|=+\s*FILE\s+(?P<b>\S+?)\s*=+)\s*$', re.M)
END_RE = re.compile(r'^(?:<<<END\s+[^>]*>>>|=+\s*END(?:\s+\S+)?\s*=+)\s*$', re.M)


def parse_blocks(out: str) -> tuple[dict, str]:
    """Robust block parser (v2, 0918). A block starts at any FILE marker, strict <<<FILE p>>> or the input-echo
    form ===== FILE p =====, and ends at the next FILE marker or at the end of the output; END markers of either
    form are stripped. Three editor deviations were observed in the pilot and are all handled: every file in the
    echo form (T-def FA0198 R2), END markers omitted between files (cand_2_2 FA0005), and the two forms mixed in
    one answer (cand_2_2 FA0002). The tag reports whether the strict-only parser would have given the same result."""
    out = out or ''
    starts = list(START_RE.finditer(out))
    blocks = {}
    for i, m in enumerate(starts):
        rel = (m.group('a') or m.group('b')).strip()
        body = out[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(out)]
        body = END_RE.sub('', body).strip('\n')
        blocks[rel] = body
    strict = {m.group('p').strip(): m.group('body').strip('\n') for m in FILE_RE.finditer(out)}
    tag = 'none' if not blocks else ('strict' if strict == blocks else 'robust')
    return blocks, tag


TEXT_SYSTEM = prompts.A1_SYSTEM   # 'You are an expert editor of machine-learning research papers written in LaTeX. You return revised LaTeX source only.'


def text_prompt(skill_txt: str, files: list, r: int, findings: str | None) -> str:
    names = ', '.join(rel for rel, _ in files)
    body = '\n'.join(f'===== FILE {rel} =====\n{txt}' for rel, txt in files)
    head = (f'A skill file, SciSlop.md, and the complete LaTeX source of a manuscript follow. Read the skill file, '
            f'then revise the manuscript as the skill file says. This is revision round {r}.\n\n'
            f'===== SciSlop.md =====\n{skill_txt}\n===== END SciSlop.md =====\n\n')
    if findings:
        head += f'===== SLOP_FINDINGS.md =====\n{findings}\n===== END SLOP_FINDINGS.md =====\n\n'
    return head + prompts.A1_FORMAT.format(path='{path}', n_files=len(files), names=names) + body + '\n\n'


def agent_prompt(r: int, loc: bool) -> str:
    s = (f'Read SciSlop.md in this directory and revise the manuscript (main.tex and sections/*.tex) as that file says. '
         f'This is revision round {r}. ')
    if loc:
        s += 'SLOP_FINDINGS.md in this directory lists the instances an external check located; treat every listed instance as SciSlop.md says. '
    s += ('Make the changes directly by editing the .tex files in place; do not just give advice. Do not edit SciSlop.md or '
          'SLOP_FINDINGS.md, do not create new files, and do not run latex. ')
    if os.environ.get('SH_EXTRA_ITEMS') == '1':
        s += ('A figure in this directory is an image file, and you can open it with the Read tool and look at it. '
              'Do that before you decide anything about that figure. ')
    s += 'When done, finish your turn without a summary.'
    return s


# ------------------------------------------------------------------ guards against M_0
NUM_RE = re.compile(r'(?<![\w.\-])\d+(?:\.\d+)?%?(?![\w.])')


def body_number_counts(d: Path) -> dict:
    """How many times each reported number appears in the body. The set-based guard sees a number as kept while it
    still occurs once, so a result dropped from three of its four mentions passes. Counting catches that. Legitimate
    repairs of recycled sentences also remove repeated numbers, so this is reported and audited, never hard."""
    from collections import Counter
    t = paper_text(d)
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\(?:cite[a-zA-Z]*|ref|eqref|label|autoref|Cref|cref|input|includegraphics|bibliography|bibliographystyle)\*?\s*(?:\[[^\]]*\])?\{[^}]*\}', ' ', t)
    return Counter(NUM_RE.findall(t))


def body_numbers(d: Path) -> set:
    t = paper_text(d)
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\(?:cite[a-zA-Z]*|ref|eqref|label|autoref|Cref|cref|input|includegraphics|bibliography|bibliographystyle)\*?\s*(?:\[[^\]]*\])?\{[^}]*\}', ' ', t)
    return set(NUM_RE.findall(t))


def labels(d: Path) -> set:
    return set(re.findall(r'\\label\{([^}]*)\}', paper_text(d)))


def refs(d: Path) -> set:
    out = set()
    for m in re.finditer(r'\\(?:ref|eqref|autoref|Cref|cref|pageref)\*?\{([^}]*)\}', paper_text(d)):
        out |= {k.strip() for k in m.group(1).split(',')}
    return out


def edit_size(m0: Path, mr: Path) -> dict:
    """Changed lines over the tex files, relative to M_0; unified diff text is returned for the trace."""
    a, b, diffs = 0, 0, []
    f0 = {str(p.relative_to(m0)): p for p in tex_files(m0)}
    fr = {str(p.relative_to(mr)): p for p in tex_files(mr)}
    n0 = 0
    for rel in sorted(set(f0) | set(fr)):
        la = f0[rel].read_text(errors='ignore').splitlines() if rel in f0 else []
        lb = fr[rel].read_text(errors='ignore').splitlines() if rel in fr else []
        n0 += len(la)
        d = list(difflib.unified_diff(la, lb, f'M0/{rel}', f'Mr/{rel}', lineterm='', n=1))
        a += sum(1 for l in d if l.startswith('+') and not l.startswith('+++'))
        b += sum(1 for l in d if l.startswith('-') and not l.startswith('---'))
        diffs += d
    return {'lines_added': a, 'lines_removed': b, 'lines_m0': n0, 'edit_ratio': round((a + b) / max(1, 2 * n0), 4),
            'diff': '\n'.join(diffs)}


def guards(m0: Path, mr: Path, scores0: dict, scoresr: dict, denom0: dict, denomr: dict) -> dict:
    c0, cr = cite_keys(m0), cite_keys(mr)
    l0, lr = labels(m0), labels(mr)
    n0, nr = body_numbers(m0), body_numbers(mr)
    cnt0, cntr = body_number_counts(m0), body_number_counts(mr)   # names must not shadow the citation-key sets above
    thinned = {k: [cnt0[k], cntr.get(k, 0)] for k in cnt0 if cntr.get(k, 0) < cnt0[k] and cnt0[k] >= 2}
    g = {'cited_works_removed': sorted(c0 - cr), 'cited_works_added': sorted(cr - c0),
         'dangling_cites': sorted(set(dangling(mr)) - set(dangling(m0))),
         'labels_removed': sorted(l0 - lr), 'refs_undefined': sorted(refs(mr) - lr),
         'numbers_lost': sorted(n0 - nr), 'numbers_added': sorted(nr - n0), 'numbers_thinned': thinned,
         'words_m0': word_count(m0), 'words_mr': word_count(mr)}
    g['word_ratio'] = round(g['words_mr'] / max(1, g['words_m0']), 3)
    shrink = {}
    for it in ITEMS:
        d0, dr = denom0.get(it), denomr.get(it)
        s0, sr = scores0.get(it), scoresr.get(it)
        if d0 and dr is not None and dr < d0 and s0 is not None and sr is not None and sr < s0:
            shrink[it] = {'denominator': [d0, dr], 'score': [s0, sr]}
    g['score_fell_with_shrinking_denominator'] = shrink
    g['evidence_gap_closed'] = bool(scores0.get('evidence_gap') == 1.0 and scoresr.get('evidence_gap') not in (None, 1.0))
    # hard = excluded from the Pareto front; soft = reported next to every result, audited by hand
    viol, soft = [], []
    if g['cited_works_removed']: viol.append('cited_works_removed')
    if g['cited_works_added']: viol.append('cited_works_added')
    if g['dangling_cites']: viol.append('dangling_cites')
    if g['refs_undefined']: viol.append('refs_undefined')
    if g['labels_removed']: soft.append('labels_removed')
    if g['numbers_lost']: viol.append('numbers_lost')
    if g['numbers_added']: soft.append('numbers_added')
    if thinned: soft.append('numbers_thinned')      # a reported number kept somewhere but dropped from other mentions          # audited: a new number may be a cross-reference or a fabricated result
    if shrink: soft.append('denominator_shrank')                 # citation denominators fall when sentences are merged (aggregate.py decision, 0915)
    if g['evidence_gap_closed']: soft.append('evidence_gap_closed_needs_audit')
    if not 0.8 <= g['word_ratio'] <= 1.25: viol.append('word_ratio')
    g['violations'] = viol; g['soft_flags'] = soft
    return g


# ------------------------------------------------------------------ measuring
# Naming an item in SH_EXTRA_SKIP drops only that extra item. To measure how much one item adds to the round time,
# run with only that item removed and everything else unchanged.
_SKIP = {x.strip() for x in os.environ.get('SH_EXTRA_SKIP', '').split(',') if x.strip()}
ACK_RE = re.compile(
    r'((no|not|never|without)\s+(any\s+)?(individual|per-example|per-instance|per-sample|single|concrete|specific|qualitative)\s+'
    r'(example|case|instance|sample|output|prediction|failure|trajector)\w*\s+(was|were|is|are|have been|has been|could be|can be)?\s*'
    r'(retained|inspected|examined|displayed|shown|reported|kept|stored|logged|saved|available|preserved|analy[sz]ed|reviewed))'
    r'|((rest|rests|relies|rely|based|draw|draws)\s+(only\s+|solely\s+|entirely\s+)?on\s+aggregate\s+(scores?|results?|metrics?|accurac\w+|numbers?))'
    r'|(aggregate\s+(scores?|results?|metrics?)\s+only)'
    r'|((did|do|does)\s+not\s+(retain|inspect|examine|store|log|save|keep|report|display)\s+(any\s+)?(individual|per-example|single|concrete|specific)\s+(example|case|instance|sample|output|prediction|failure)s?)',
    re.I)
EXTRA_ORDER = [x for x in ('argument_graph', 'fig_exposition') if x not in _SKIP]


def measure(code: str, tree: Path, out: Path) -> dict:
    """Runs the deterministic measurers on the tree; returns units, scores, denominators.

    With SH_EXTRA_ITEMS=1 the argument graph and the figure exposition are measured too. The first needs the
    labelling server and a GPU and is re-measured every round because the introduction changes. The second is an
    image item whose value cannot move, so its original value is carried and only the figure's presence is checked."""
    t0 = time.time()
    r = failing_units(code, tree, out, ITEMS)
    den = {}
    for it in ITEMS:
        p = out / it / 'papers.jsonl'
        if p.exists():
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if rows:
                den[it] = rows[0].get('slop_denominator')
    r['denominators'] = den
    if os.environ.get('SH_EG_ACK', '1') == '1' and r['scores'].get('evidence_gap') == 1.0:
        # evidence gap v2 (0920). The second repair when no specimen exists = stating the gap. If the manuscript says that individual cases were
        # not kept or inspected and the claim rests on aggregates only, it is scored as a disclosed gap (0.5) rather than a silent gap (1.0). The measurer
        # code is unchanged; the harness judges on top of it. The human-pair value is the raw measurer value.
        if ACK_RE.search(re.sub(r'\s+', ' ', paper_text(Path(tree)))):
            r['scores']['evidence_gap'] = 0.5
            r['units']['evidence_gap'] = []
            r.setdefault('extra', {})['evidence_gap'] = {'verdict': 'acknowledged', 'note': 'the manuscript states that no individual case was retained or inspected; the gap is explicit, not silent'}
        else:
            r.setdefault('extra', {})['evidence_gap'] = {'verdict': 'silent'}
    if os.environ.get('SH_EXTRA_ITEMS') == '1':
        sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
        import extra_items as X
        if 'argument_graph' in EXTRA_ORDER:
            ag = X.argument_graph_measure(code, tree, Path(out) / 'argument_graph')
            r['scores']['argument_graph'] = ag.get('slop_score')
            r['units']['argument_graph'] = X.argument_graph_units(code, Path(out) / 'argument_graph')
            r.setdefault('extra', {})['argument_graph'] = ag
        if 'fig_exposition' in EXTRA_ORDER:
            fe = X.fig_exposition_score(code, Path(tree))
            r['scores']['fig_exposition'] = fe.get('slop_score')
            r['units']['fig_exposition'] = X.fig_exposition_units(code, Path(tree))
            r.setdefault('extra', {})['fig_exposition'] = fe
    r['measure_s'] = round(time.time() - t0, 1)
    json.dump({k: v for k, v in r.items()}, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False)
    return r


# ------------------------------------------------------------------ one round
def one_round(arm: str, code: str, r: int, src: Path, dst: Path, skill_path: Path, units: dict | None, logdir: Path,
              model: str = EDITOR, notes: str | None = None) -> dict:
    attach, mode = arm.split('-')            # T|A, def|loc
    loc = mode == 'loc'
    stage(src, dst)
    spec = None
    if os.environ.get('SH_EXTRA_ITEMS') == '1':
        sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
        import extra_items as _X
        # The diagram is carried over from the previous round's tree (edited figure and EDIT_REPORT). For r = 1, src is the original, so the original comes.
        if 'fig_exposition' in EXTRA_ORDER:
            _X.stage_method_figure(code, dst, prev=src)
        if attach == 'A' and os.environ.get('SH_SPECIMEN', '1') == '1':
            # The editor must be able to see the records for the evidence-gap fix to work. The specimen search runs once per paper and is cached.
            spec = _X.ensure_specimen(code)
            _X.stage_editor_materials(code, dst, spec)
    skill_txt = Path(skill_path).read_text()
    cap = location_cap(skill_path)
    findings = findings_text(units, cap, r, mode=location_mode(skill_path)) if (loc and units is not None) else None
    if loc and findings is None:
        return {'exec': 'nothing_to_fix'}
    if findings and spec is not None and (units or {}).get('evidence_gap'):
        findings += _X.specimen_note(spec)
    if notes and attach == 'T':
        findings = (findings or '') + '\n' + notes
    md5_before = tex_md5(dst)
    rec = {'arm': arm, 'code': code, 'round': r, 'model': model, 'skill_md5': hashlib.md5(skill_txt.encode()).hexdigest(),
           'location_cap': cap, 'n_units_given': {k: min(len(v), cap) if k != 'evidence_gap' else len(v) for k, v in (units or {}).items()} if loc else None}
    files = a1_files(dst)
    payload = [(str(f.relative_to(dst)), f.read_text(errors='ignore')) for f in files]
    if attach == 'T' and loc and calls_per_round(skill_path) == 'per_entry':
        # one editor turn per entry with instances, in skill order; the tree is updated between turns
        rec['policy'] = 'per_entry'; calls = []; acc_all = {}
        prompt_all = []
        for it in ['xsec_ref', 'macro_redund', 'citation', 'evidence_gap']:
            f_it = findings_text(units, cap, r, only=it, mode=location_mode(skill_path))
            if not f_it:
                continue
            files = a1_files(dst); payload = [(str(f.relative_to(dst)), f.read_text(errors='ignore')) for f in files]
            prompt = text_prompt(skill_txt, payload, r, f_it)
            res = run_text_json(prompt, TEXT_SYSTEM, dst, model)
            prompt_all.append(f'##### CALL {it}\n' + prompt)
            blocks, _parser = parse_blocks(res.get('result') or '')
            acc = {}
            for rel, orig in payload:
                ok, why = _accept(orig, blocks.get(rel), rel)
                if ok:
                    (dst / rel).write_text(blocks[rel]); acc[rel] = 'revised'
                else:
                    acc[rel] = f'kept_original:{why}'
            calls.append({'entry': it, 'files': acc, 'n_blocks_returned': len(blocks),
                          'call': {k: res.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'duration_ms', 'num_turns', 'usage', 'is_error', 'stop_reason', 'limited')}})
            acc_all[it] = acc
            if res.get('limited'):
                break
        prompt = '\n\n'.join(prompt_all)
        rec['files'] = acc_all; rec['calls'] = calls; rec['prompt_chars'] = len(prompt)
        res = {'result': '\n\n'.join(f'##### CALL {c["entry"]}\n' for c in calls), 'stderr': '',
               'rc': max(c['call']['rc'] for c in calls), 'wall_s': round(sum(c['call']['wall_s'] for c in calls), 1),
               'cost_usd': sum(c['call'].get('cost_usd') or 0 for c in calls), 'duration_ms': sum(c['call'].get('duration_ms') or 0 for c in calls),
               'num_turns': sum(c['call'].get('num_turns') or 0 for c in calls), 'limited': any(c['call'].get('limited') for c in calls),
               'usage': {k: sum((c['call'].get('usage') or {}).get(k, 0) for c in calls) for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')}}
    elif attach == 'T':
        prompt = text_prompt(skill_txt, payload, r, findings)
        rec['prompt_chars'] = len(prompt)
        res = run_text_json(prompt, TEXT_SYSTEM, dst, model)
        blocks, rec['parser'] = parse_blocks(res.get('result') or '')
        acc = {}
        for rel, orig in payload:
            ok, why = _accept(orig, blocks.get(rel), rel)
            if ok:
                (dst / rel).write_text(blocks[rel]); acc[rel] = 'revised'
            else:
                acc[rel] = f'kept_original:{why}'
        rec['files'] = acc; rec['n_blocks_returned'] = len(blocks)
    else:
        (dst / 'SciSlop.md').write_text(skill_txt)
        if findings:
            (dst / 'SLOP_FINDINGS.md').write_text(findings)
        prompt = agent_prompt(r, loc)
        if notes:
            (dst / 'REVIEW_NOTES.md').write_text(notes)
            prompt += ' REVIEW_NOTES.md in this directory lists changes of the previous round that a reviewer reverted, with reasons; read it first and do not repeat them.'
        rec['prompt_chars'] = len(prompt)
        mode = os.environ.get('SH_EDITOR_MODE', 'agent')
        if mode == 'text_parallel' and loc:
            res = run_text_parallel(skill_txt, findings, notes, r, dst, model, code, rec)
        elif mode == 'edits_parallel' and (loc or os.environ.get('SH_EDITS_PARALLEL_DEF') == '1'):
            # SH_EDITS_PARALLEL_DEF=1 (0922 ablation): the location-less A-def arm also runs with the same per-file edit-block attachment
            res = run_edits_parallel(skill_txt, findings, notes, r, dst, model, code, rec, units=units)
        elif (mode == 'agent_parallel' or os.environ.get('SH_EDITOR_PARALLEL', '0') == '1') and loc:
            res = run_agents_parallel(prompt, dst, model, code, rec)
        else:
            res = run_agent_json(prompt, dst, model)
        rec['skill_file_edited_by_agent'] = (dst / 'SciSlop.md').read_text() != skill_txt
        if os.environ.get('SH_EXTRA_ITEMS') == '1':
            sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
            import extra_items as _X2
            er = _X2.apply_erase_request(code, dst) if 'fig_exposition' in EXTRA_ORDER else None
            if er:
                rec['figure_edit'] = er
        extra = sorted(str(p.relative_to(dst)) for p in dst.rglob('*') if p.is_file()
                       and p.suffix not in ('.tex', '.bib', '.bst', '.sty', '.jpeg', '.jpg', '.png', '.pdf')
                       and p.name not in ('SciSlop.md', 'SLOP_FINDINGS.md', 'REVIEW_NOTES.md', 'ERASE.txt', 'EDIT_REPORT.json')
                       and not str(p.relative_to(dst)).startswith('materials/')
                       and '/_bands/' not in '/' + str(p.relative_to(dst)))
        rec['files_created_by_agent'] = extra
    rec['call'] = {k: res.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'duration_ms', 'duration_api_ms', 'num_turns', 'usage', 'model_usage', 'is_error', 'stop_reason', 'subtype', 'limited')}
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f'R{r}_prompt.txt').write_text(prompt)
    (logdir / f'R{r}_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    io = rec.pop('_per_file_io', None)
    if io:
        # The parallel editor gets a different instruction per file. Keep the full text it received and returned, per file.
        ed = logdir / f'R{r}_editor'
        ed.mkdir(exist_ok=True)
        for rel, (pp, out) in io.items():
            stem = rel.replace('/', '__')
            (ed / f'{stem}.prompt.txt').write_text(pp)
            (ed / f'{stem}.stdout.txt').write_text(out)
    if findings:
        (logdir / f'R{r}_findings.md').write_text(findings)
    rec['md5_before'], rec['md5_after'] = md5_before, tex_md5(dst)
    rec['exec'] = 'limited' if res.get('limited') else ('ok' if rec['md5_after'] != md5_before else 'noop')
    return rec


def snapshot_round(base: Path, code: str, r: int, tree: Path, traj: dict) -> None:
    """Collect the per-round outputs into base/rounds/R<r>/ and rewrite the base/ROUNDS.md table.

    What is kept: that round's method-diagram image, ERASE.txt, EDIT_REPORT.json, and the score/guard/gate summary (scores.json).
    The round tree and measure/ gate/ logs/ stay as they are, so this is the summary an analysis opens first."""
    try:
        out = base / 'rounds' / f'R{r}'
        out.mkdir(parents=True, exist_ok=True)
        if os.environ.get('SH_EXTRA_ITEMS') == '1':
            sys.path.insert(0, str(TEMP.parent / 'api' / 'code'))
            import extra_items as _X
            fig = _X.method_figure(code)
            if fig:
                name = Path(fig).name
                for cand in (Path(tree) / 'figures' / name, Path(tree) / fig):
                    if cand.exists():
                        shutil.copy2(cand, out / f'figure{cand.suffix}'); break
            for extra in ('ERASE.txt', 'EDIT_REPORT.json'):
                f = Path(tree) / 'figures' / extra
                if f.exists():
                    shutil.copy2(f, out / extra)
        if r == 0:
            rec = {'round': 0, 'scores': traj['r0']['scores'], 'n_units': traj['r0'].get('n_units')}
        else:
            x = traj['rounds'][-1]
            rec = {'round': r, 'exec': x.get('exec'), 'scores': x.get('scores_after'), 'n_units': x.get('n_units_after'),
                   'guards': {'violations': x['guards']['violations'], 'soft_flags': x['guards']['soft_flags']},
                   'gate': {k: (x.get('gate') or {}).get(k) for k in ('kept', 'reverted', 'retired_total')},
                   'figure_edit': {k: (x.get('figure_edit') or {}).get(k) for k in ('status', 'file_changed', 'applied')},
                   'editor_call': {k: (x.get('call') or {}).get(k) for k in ('wall_s', 'cost_usd', 'num_turns')},
                   'measure_s': x.get('measure_s')}
        (out / 'scores.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        if r > 0:
            # Put the round's inputs and verdicts together so an analysis can start from one folder. The originals stay in the tree and in gate/ logs/.
            for src, name in ((Path(tree) / 'SLOP_FINDINGS.md', 'SLOP_FINDINGS.md'), (Path(tree) / 'REVIEW_NOTES.md', 'REVIEW_NOTES_in.md'),
                              (base / 'gate' / f'R{r}' / 'VERDICTS.json', 'VERDICTS.json'), (base / 'gate' / f'R{r}' / 'RETIRE.json', 'RETIRE.json'),
                              (base / 'gate' / f'R{r}' / 'CHANGES.md', 'CHANGES.md'), (base / 'measure' / f'R{r}' / 'summary.json', 'measure_summary.json'),
                              (base / 'logs' / f'R{r}_diff_vs_prev.patch', 'diff_vs_prev.patch'), (base / 'logs' / f'R{r}_diff_vs_M0.patch', 'diff_vs_M0.patch')):
                if src.exists():
                    shutil.copy2(src, out / name)
            (out / 'README.md').write_text(
                f'# R{r} output guide\n\n'
                f'- Final manuscript of this round (after the gate) = `../../R{r}/` (tex, figures/, materials/)\n'
                f'- Pre-gate manuscript (as the editor returned it) = `../../gate/R{r}/REVISED/`, previous round = `../../gate/R{r}/PREVIOUS/`, original = `../../gate/R{r}/ORIGINAL/`\n'
                f'- Editor input/output = `../../logs/R{r}_prompt.txt`, `R{r}_stdout.txt`, per file `R{r}_editor/`\n'
                f'- Reviewer input/output = `../../gate/R{r}/chunk_k/`, `retire_stdout.txt`\n'
                f'- Measurement details = `../../measure/R{r}/<item>/`\n'
                f'- This folder = diagram image, ERASE.txt, EDIT_REPORT.json, scores.json, SLOP_FINDINGS.md, REVIEW_NOTES_in.md, VERDICTS.json, RETIRE.json, CHANGES.md, measure_summary.json, diff_vs_prev.patch, diff_vs_M0.patch\n')
        items = list(traj['r0']['scores'].keys())
        rows = [('R0', traj['r0']['scores'], None)] + [(f"R{x['round']}", x.get('scores_after') or {}, x) for x in traj['rounds']]
        md = [f'# {code} per-round record', '',
              '| Round | ' + ' | '.join(items) + ' | Exec | Gate kept/reverted | Diagram edit | Hard guards |',
              '|---' * (len(items) + 5) + '|']
        fmt = lambda v: '.' if v is None else f'{v:.3f}'
        for name, sc, x in rows:
            g = (x or {}).get('gate') or {}
            fe = (x or {}).get('figure_edit') or {}
            hard = ', '.join(((x or {}).get('guards') or {}).get('violations') or []) or ('' if x is None else 'none')
            md.append(f'| {name} | ' + ' | '.join(fmt(sc.get(i)) for i in items) +
                      f" | {(x or {}).get('exec', 'original')} | {g.get('kept', '')}/{g.get('reverted', '')} | "
                      f"{fe.get('status') or ''} | {hard} |")
        (base / 'ROUNDS.md').write_text('\n'.join(md) + '\n')
    except Exception as e:
        (base / f'snapshot_error_R{r}.txt').write_text(repr(e))


def run_paper(arm: str, code: str, skill_path: Path, runs: Path, r0_measure: dict, r0_dir: Path, model: str = EDITOR,
              round_limit: int = ROUND_LIMIT, log=print, resume: dict | None = None) -> dict:
    """Recursive Slop Mitigation on one manuscript. Returns the trajectory record.
    resume = {'traj': existing trajectory with completed ok rounds, 'prev_dir', 'prev_measure', 'start_round'} continues it."""
    base = runs / arm / code
    logdir = base / 'logs'
    traj = {'arm': arm, 'code': code, 'skill': str(skill_path), 'model': model, 'rounds': [], 'r0': {'scores': r0_measure['scores'], 'n_units': {k: len(v) for k, v in r0_measure['units'].items()}}}
    m0 = fars_paper_dir(code)
    prev_dir, prev_measure = m0, r0_measure
    notes = None
    retired: list = []
    start = 1
    if resume:
        traj = resume['traj']; prev_dir, prev_measure, start = Path(resume['prev_dir']), resume['prev_measure'], resume['start_round']
    t_start = time.time()
    stop = None
    if not resume:
        snapshot_round(base, code, 0, m0, traj)
    for r in range(start, round_limit + 1):
        units = prev_measure['units']
        if retired:
            import quality_gate as Q
            units = Q.drop_retired(units, {}, retired)
        if arm.endswith('loc') and r > 1 and not any(units.get(it) for it in units):
            stop = 'no_remaining_units'; break
        dst = base / f'R{r}'
        rec = None
        # A usage limit either clears within minutes or stays for a long time. Waiting eight times 10 minutes would waste
        # 80 minutes doing nothing, so check briefly twice and give up (this lost 83 minutes on 0919 evening).
        tries = int(os.environ.get('SH_LIMIT_RETRIES', '2'))
        nap = int(os.environ.get('SH_LIMIT_SLEEP', '180'))
        for attempt in range(tries + 1):
            rec = one_round(arm, code, r, prev_dir, dst, skill_path, units if arm.endswith('loc') else None, logdir, model, notes=notes)
            if rec.get('exec') == 'limited' and attempt < tries:
                log(f'[{arm} {code} R{r}] session limit, sleeping {nap}s ({attempt + 1}/{tries})'); time.sleep(nap); continue
            break
        if rec.get('exec') == 'limited':
            stop = 'session_limit'; log(f'[{arm} {code} R{r}] session limit, giving up'); break
        if rec.get('exec') == 'nothing_to_fix':
            stop = 'no_remaining_units'; break
        if rec['exec'] == 'ok' and os.environ.get('SH_GATE') == '1':
            import quality_gate as Q
            # 0922 component ablation. To reduce only the editor's information while holding the reviewer fixed, the reviewer always gets the full skill file
            # (SH_GATE_SKILL) and the full location list (SH_GATE_FULL_FINDINGS=1; if absent from the editor tree, rebuilt from this round's measurement).
            gate_skill = Path(os.environ['SH_GATE_SKILL']) if os.environ.get('SH_GATE_SKILL') else skill_path
            if os.environ.get('SH_GATE_FULL_FINDINGS') == '1' and not (dst / 'SLOP_FINDINGS.md').exists():
                full_f = findings_text(units, location_cap(gate_skill), r, mode=location_mode(gate_skill))
                if full_f:
                    (dst / 'SLOP_FINDINGS.md').write_text(full_f)
                    rec['gate_findings'] = 'written_for_gate_only'
            g_rec = Q.review(m0, prev_dir, dst, code, gate_skill, base / 'gate' / f'R{r}')
            applied = Q.apply_verdicts(prev_dir, dst, g_rec)
            notes = Q.notes_for_editor(g_rec)
            retired = retired + (g_rec.get('retire') or [])
            rec['gate'] = {k: v for k, v in g_rec.items() if k != '_changes'}; rec['gate'].update(applied)
            rec['gate']['retired_total'] = len(retired)
            rec['md5_after_gate'] = tex_md5(dst)
            if rec['md5_after_gate'] == rec['md5_before']:
                rec['exec'] = 'gated_noop'
            log(f'[{arm} {code} R{r}] gate kept {applied["kept"]} reverted {applied["reverted"]} '
                f'retired {len(g_rec.get("retire") or [])} '
                f'(${(g_rec["call"] or {}).get("cost_usd") or 0:.2f}, {(g_rec["call"] or {}).get("wall_s")}s)')
        if rec['exec'] in ('ok', 'gated_noop'):
            meas = measure(code, dst, base / 'measure' / f'R{r}') if rec['exec'] == 'ok' else prev_measure
        else:
            meas = prev_measure
        es = edit_size(m0, dst)
        (logdir / f'R{r}_diff_vs_M0.patch').write_text(es.pop('diff'))
        try:
            dp = subprocess.run(['diff', '-ru', '--exclude=figures', '--exclude=materials', '--exclude=*.md', '--exclude=_bands',
                                 str(prev_dir), str(dst)], capture_output=True, text=True).stdout
            (logdir / f'R{r}_diff_vs_prev.patch').write_text(dp)
        except Exception:
            pass
        g = guards(m0, dst, r0_measure['scores'], meas['scores'], r0_measure['denominators'], meas['denominators'])
        rec.update({'scores_after': meas['scores'], 'n_units_after': {k: len(v) for k, v in meas['units'].items()},
                    'denominators_after': meas['denominators'], 'measure_s': meas.get('measure_s'), 'edit_size_vs_m0': es, 'guards': g})
        traj['rounds'].append(rec)
        log(f'[{arm} {code} R{r}] {rec["exec"]} wall={rec["call"]["wall_s"]}s cost=${rec["call"].get("cost_usd") or 0:.3f} '
            f'turns={rec["call"].get("num_turns")} scores={meas["scores"]} guards={g["violations"]} soft={g["soft_flags"]}')
        # Record after every round. Analysis must be possible even if the run dies midway, and the diagram must be kept per round.
        traj['stop'] = 'in_progress'
        json.dump(traj, open(base / 'trajectory.json', 'w'), indent=1, ensure_ascii=False)
        snapshot_round(base, code, r, dst, traj)
        if rec['exec'] == 'gated_noop':
            if r > 1 and traj['rounds'][-2].get('exec') == 'gated_noop':
                stop = 'gated_noop_twice'; break
            prev_dir, prev_measure = dst, meas   # tree equals previous round after reverts; editor gets the notes next round
            continue
        if rec['exec'] != 'ok':
            stop = 'no_op'; break
        prev_dir, prev_measure = dst, meas
    traj['stop'] = stop or 'round_limit'
    traj['wall_total_s'] = round(time.time() - t_start + (traj.get('wall_total_s', 0) if resume else 0), 1)
    traj['cost_total_usd'] = round(sum((x['call'].get('cost_usd') or 0) for x in traj['rounds']), 4)
    traj['final_dir'] = str(prev_dir)
    traj['final_scores'] = prev_measure['scores']
    json.dump(traj, open(base / 'trajectory.json', 'w'), indent=1, ensure_ascii=False)
    return traj


def r0(code: str, runs: Path) -> tuple[dict, Path]:
    d = runs / '_R0' / code
    s = d / 'measure' / 'summary.json'
    if s.exists():
        return json.load(open(s)), fars_paper_dir(code)
    return measure(code, fars_paper_dir(code), d / 'measure'), fars_paper_dir(code)
