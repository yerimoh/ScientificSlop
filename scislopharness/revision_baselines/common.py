"""Shared paths, staging, no-op detection, Claude CLI runner, progress records.
Effects_of_revision (0914). See ../DESIGN.md."""
from __future__ import annotations
import glob, hashlib, json, os, re, shutil, subprocess, time, urllib.error
from pathlib import Path

ROOT = Path(os.environ.get("SCISLOP_ROOT", "."))
EOR = ROOT / 'paper/draft_v6/Effects_of_revision'
BENCH = ROOT / 'paper/draft_v6/scislopbench/bench165'
SLOP = ROOT / 'paper/draft_v6/slop'
FARS = ROOT / 'fars/papers'
B3_RUNS = ROOT / 'artifact-ai2science/Evaluation/02_baselines_B/B3_reviewer_loop/runs'
LLM_DIR = ROOT / 'artifact-ai2science/_llm'
CLAUDE = Path.home() / '.local/bin/claude'
MODEL = os.environ.get('EOR_MODEL', 'claude-haiku-4-5-20251001')   # editor model; override for model variants
TAG = os.environ.get('EOR_TAG', '')                                      # storage suffix, e.g. '.sonnet'
HERE = Path(__file__).resolve().parent
AGENT_SETTINGS = HERE / 'agent_settings.json'
ROUNDS = 3
ITEMS = ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap']

_ITEMS165 = None


def items165():
    global _ITEMS165
    if _ITEMS165 is None:
        _ITEMS165 = json.load(open(BENCH / 'items165.json'))['items']
    return _ITEMS165


def ai_codes() -> list[str]:
    return sorted(i['pair'] for i in items165() if i['label'] == 1)


def hu_record(code: str) -> dict | None:
    for i in items165():
        if i['label'] == 0 and i['pair'] == code:
            main = os.path.join(i['tex_dir'], i['tex_root'])
            return {'corpus': 'HU', 'id': i['arxiv'], 'root': i['tex_dir'], 'main_tex': main,
                    'paper_dir': i['tex_dir'], 'exp_dir': None, 'diagram': None,
                    'anchor_of': code, 'anchor_year': None, 'anchor_venue': i.get('venue')}
    return None


def fars_paper_dir(code: str) -> Path:
    g = glob.glob(str(FARS / f'{code}_*/code/writing/paper'))
    assert g and (Path(g[0]) / 'main.tex').exists(), code
    return Path(g[0])


def ai_record(code: str, d: Path) -> dict:
    return {'corpus': 'AI', 'id': code, 'root': str(d), 'main_tex': str(d / 'main.tex'),
            'paper_dir': str(d), 'exp_dir': None, 'diagram': None}


# ------------------------------------------------------------------ staging
STAGE_EXT = ('.tex', '.bib', '.bst', '.sty')


def stage(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in Path(src).iterdir():
        if f.is_file() and f.suffix in STAGE_EXT:
            shutil.copy2(f, dst / f.name)
    if (Path(src) / 'sections').is_dir():
        (dst / 'sections').mkdir()
        for f in (Path(src) / 'sections').iterdir():
            if f.is_file() and f.suffix == '.tex':
                shutil.copy2(f, dst / 'sections' / f.name)
    return dst


def tex_files(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob('*.tex') if p.name != 'math_commands.tex')


def tex_md5(d: Path) -> str | None:
    m = hashlib.md5(); n = 0
    for f in tex_files(d):
        m.update(f.read_bytes()); n += 1
    return m.hexdigest() if n else None


def file_md5s(d: Path) -> dict:
    return {str(f.relative_to(d)): hashlib.md5(f.read_bytes()).hexdigest() for f in tex_files(d)}


def paper_text(d: Path) -> str:
    t = (Path(d) / 'main.tex').read_text(errors='ignore')
    for f in sorted((Path(d) / 'sections').glob('*.tex')) if (Path(d) / 'sections').is_dir() else []:
        t += '\n' + f.read_text(errors='ignore')
    return t


_WORD = re.compile(r'[A-Za-z]{2,}')


def word_count(d: Path) -> int:
    t = paper_text(d)
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\[a-zA-Z@]+', ' ', t)
    return len(_WORD.findall(t))


CITE_RE = re.compile(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}')


def cite_keys(d: Path) -> set:
    keys = set()
    for f in tex_files(d):
        for m in CITE_RE.finditer(f.read_text(errors='ignore')):
            keys |= {k.strip() for k in m.group(1).split(',') if k.strip()}
    return keys


def bib_keys(d: Path) -> set:
    keys = set()
    for f in Path(d).glob('*.bib'):
        keys |= set(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', f.read_text(errors='ignore')))
    return keys


def dangling(d: Path) -> list[str]:
    return sorted(cite_keys(d) - bib_keys(d))


# ------------------------------------------------------------------ Claude CLI
_STRIP_ENV_PREFIX = ('CLAUDECODE', 'CLAUDE_CODE_', 'CLAUDE_PID', 'CLAUDE_EFFORT', 'CLAUDE_AGENT_SDK', 'AI_AGENT')


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_STRIP_ENV_PREFIX)}
    env['PATH'] = f"{Path.home()}/.local/bin:" + env.get('PATH', '')
    return env


SESSION_LIMIT_RE = re.compile(r"hit your (session|usage) limit|rate limit|Claude usage limit reached|"
                              r"disabled Claude subscription access|Use an Anthropic API key|"
                              r"overloaded|not available|authentication|unauthori[sz]ed|Too Many Requests", re.I)


def is_session_limit(text: str | None) -> bool:
    return bool(text) and bool(SESSION_LIMIT_RE.search(text))


def run_cli(args: list[str], cwd: Path, timeout: int = 1500) -> tuple[int, str, str, float]:
    t0 = time.time()
    try:
        r = subprocess.run([str(CLAUDE)] + args, cwd=str(cwd), env=clean_env(),
                           capture_output=True, text=True, timeout=timeout)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        rc, so, se = -1, '', 'TIMEOUT'
    return rc, so, se, time.time() - t0


def write_exec_log(logdir: Path, tag: str, prompt: str, rc, so, se, dt, note=''):
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f'{tag}_exec.txt').write_text(
        f'PROMPT:\n{prompt}\n\n=== exit={rc} dt={dt:.0f}s {note} ===\nSTDOUT:\n{so}\n\nSTDERR:\n{se[:3000]}')


def run_agent(rdir: Path, prompt: str, tag: str, logdir: Path, system: str | None,
              restricted: bool = False, note: str = '') -> tuple[bool, float]:
    """File-editing agent over rdir. system=None -> stock Claude Code system prompt."""
    args = ['-p', prompt, '--model', MODEL, '--permission-mode', 'acceptEdits',
            '--add-dir', str(rdir), '--disallowed-tools', 'WebSearch,WebFetch', '--output-format', 'text']
    if system is not None:
        args += ['--system-prompt', system, '--settings', str(AGENT_SETTINGS)]
    if restricted:
        args += ['--restricted']
    rc, so, se, dt = run_cli(args, rdir)
    write_exec_log(logdir, tag, prompt, rc, so, se, dt, note)
    return rc == 0, dt, (so or '') + (se or '')


BACKEND = os.environ.get('EOR_BACKEND', 'claude')       # 'claude' (CLI) or 'vllm' (OpenAI-compatible server, open-model pilot 0916)
VLLM_ENDPOINT_FILE = LLM_DIR / 'llm_endpoint_qwen_edit.txt'


def run_text_vllm(prompt: str, system: str, tag: str, logdir: Path, max_tokens: int = 16000, timeout: int = 2400):
    """Same text-in / text-out call as run_text, against a vLLM OpenAI-compatible server (greedy).
    Returns (text or None, dt, raw). A refusal by length (context overflow) comes back as None with the error in raw."""
    import urllib.request
    ep = os.environ.get('LLM_ENDPOINT') or VLLM_ENDPOINT_FILE.read_text().strip()
    # Decoding (0916 smoke test): greedy Qwen2.5-32B collapsed into token salad after the first file on FA0001, twice,
    # identically, with and without prefix caching. Mild sampling with a fixed seed and a light repetition penalty is
    # the standard remedy; the Claude arms sample at the CLI default too, so this does not add a protocol difference.
    dec = {'temperature': float(os.environ.get('EOR_TEMP', '0.7')), 'top_p': float(os.environ.get('EOR_TOP_P', '0.95')),
           'repetition_penalty': float(os.environ.get('EOR_REP_PEN', '1.05')),
           'seed': int(os.environ.get('EOR_SEED', '914')) + int(hashlib.md5(tag.encode()).hexdigest()[:6], 16)}   # fixed per (paper, file, attempt), different across attempts
    body = json.dumps({'model': 'qwen', 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, **dec}).encode()
    t0 = time.time(); so, se, rc = '', '', 0
    for attempt in range(3):
        try:
            req = urllib.request.Request(f'{ep.rstrip("/")}/chat/completions', data=body, headers={'Content-Type': 'application/json'})
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            so = r['choices'][0]['message']['content'] or ''; se = json.dumps({k: r.get(k) for k in ('usage',)}) + ' finish=' + str(r['choices'][0].get('finish_reason'))
            break
        except urllib.error.HTTPError as e:      # 400 = prompt too long for the context: do not retry
            se = f'HTTP {e.code} {e.read()[:500]!r}'; rc = e.code
            if e.code == 400:
                break
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            se = f'__ERR__{e!r}'; rc = -1; time.sleep(10 * (attempt + 1))
    dt = time.time() - t0
    write_exec_log(logdir, tag, prompt, rc, so[:20000], se, dt, note=f'backend=vllm model={MODEL} decode={dec}')
    return (so if so else None), dt, (so or '') + (se or '')


def run_text(prompt: str, system: str, cwd: Path, tag: str, logdir: Path) -> tuple[str | None, float]:
    """Tool-free text call (arm a1_base). Returns stdout or None."""
    if BACKEND == 'vllm':
        return run_text_vllm(prompt, system, tag, logdir)
    args = ['-p', prompt, '--model', MODEL, '--tools', '', '--system-prompt', system, '--output-format', 'text']
    rc, so, se, dt = run_cli(args, cwd, timeout=900)
    write_exec_log(logdir, tag, prompt, rc, so[:20000], se, dt)
    return (so if rc == 0 else None), dt, (so or '') + (se or '')


# ------------------------------------------------------------------ progress
def key(arm: str) -> str:
    return arm + TAG


def progress_path(arm: str, shard: int | None = None) -> Path:
    (EOR / 'progress').mkdir(exist_ok=True)
    a = key(arm)
    return EOR / 'progress' / (f'{a}.jsonl' if shard is None else f'{a}.shard{shard}.jsonl')


def load_progress(arm: str) -> dict:
    """{code: {round: record}} over every shard file of the arm."""
    out: dict = {}
    for p in sorted((EOR / 'progress').glob(f'{key(arm)}.shard*.jsonl')) + [progress_path(arm)]:
        if not p.exists():
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cur = out.setdefault(r['code'], {}).get(r['round'])
            if cur is None or r.get('ts', '') >= cur.get('ts', ''):
                out[r['code']][r['round']] = r
    return out


def append_progress(arm: str, shard: int | None, rec: dict):
    with open(progress_path(arm, shard), 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def runs_dir(arm: str) -> Path:
    return EOR / 'runs' / key(arm)


def round_dir(arm: str, code: str, n: int) -> Path:
    return runs_dir(arm) / code / f'R{n}'
