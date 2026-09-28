"""LaTeX table parser for claim_table and eval_surface.

Design goals taken from the failure record of the old CEI detector (1/165 recall) and the
Haiku-judged Table_Text_Agreement (reversed, rejected):
  * parse EVERY tabular in the document, not just \\begin{table} blocks;
  * expand \\multicolumn into spanned cells and carry \\multirow group labels down;
  * keep the printed string of each number (rounding intervals are compared at printed
    precision; absolute tolerances are forbidden because 0.005 matched 84% of random pairs);
  * read "12.3 +- 0.4", "12.3 (0.4)", "12.3_{+-0.4}", bold, percent, dagger marks;
  * detect metric direction (higher/lower better) from header arrows and lexicon, else None;
  * every cell keeps (row_index, col_index) so a locus can be written as (table, row, col).
"""
from __future__ import annotations
import re

LOWER_BETTER = re.compile(r"\b(nll|loss|error|err|brier|ece|rmse|mae|mse|perplexity|ppl|fid|wer|cer|regret|latency|runtime|"
                          r"time|cost|variance|deviation|distance|calibration|fpr|fnr|asr|attack success|misaligned|mis|"
                          r"incoherent|inc|toxicity|violation|gap|delta)\b", re.I)
HIGHER_BETTER = re.compile(r"\b(accuracy|acc|f1|auc|auroc|auprc|precision|recall|ndcg|r@\d+|recall@\d+|map|bleu|rouge|"
                           r"score|iou|dice|success|reward|correlation|spearman|pearson|hit@\d+|mrr|pass@\d+|em|exact match|"
                           r"win rate|coverage|robustness|sharpe|return|ir|calmar)\b", re.I)

TABLE_ENV_RE = re.compile(r"\\begin\{(table\*?|sidewaystable\*?|wraptable)\}(.*?)\\end\{\1\}", re.S)
TABULAR_RE = re.compile(r"\\begin\{(tabular\*?|tabularx|longtable|tabulary)\}\s*(?:\{[^{}]*\}\s*)?(?:\{[^{}]*\})?\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\})?(.*?)\\end\{\1\}", re.S)
CAPTION_RE = re.compile(r"\\caption\*?\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.S)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
NUM_RE = re.compile(r"(?<![\w.])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)(?![\w.])")


def clean_cell(c: str) -> str:
    c = re.sub(r"\\(?:textbf|mathbf|bm|textit|emph|underline|textsc|texttt|text|mathrm|textcolor\{[^}]*\})\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", c)
    c = re.sub(r"\\(?:cellcolor|rowcolor|color)\s*(?:\[[^\]]*\])?\{[^}]*\}", " ", c)
    c = re.sub(r"\\(?:phantom|hphantom)\{[^}]*\}", " ", c)
    c = c.replace("\\%", "%").replace("\\&", "&").replace("~", " ")
    c = re.sub(r"\\[a-zA-Z]+\*?", " ", c)
    c = re.sub(r"[{}$]", " ", c)
    return re.sub(r"\s+", " ", c).strip()


def parse_number_cell(raw: str) -> dict:
    """Return {"kind": "number", "value", "printed", "std", "std_printed", "is_bold", "is_percent"}
    or {"kind": "text", ...}."""
    is_bold = bool(re.search(r"\\(?:textbf|mathbf|bm)\b", raw))
    s = raw
    is_percent = "\\%" in s or "%" in s
    std = None
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:\\pm|±|\$\\pm\$|\\textpm|\+/-|\+-)\s*\{?\s*([-+]?\d+(?:\.\d+)?)", s)
    if m:
        val, std = m.group(1), m.group(2)
    else:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*_\{\s*\\pm\s*([-+]?\d+(?:\.\d+)?)\s*\}", s)
        if m:
            val, std = m.group(1), m.group(2)
        else:
            c = clean_cell(s)
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*\)", c)
            if m:
                val, std = m.group(1), m.group(2)
            else:
                c2 = re.sub(r"\[[^\]]*\]", " ", c)
                nums = NUM_RE.findall(c2.replace(",", ""))
                residue = re.sub(r"\b(?:k|M|B|ms|s|x|X|pp|std|avg)\b", "", c2)
                if len(nums) == 1 and not re.search(r"[A-Za-z]{3,}", residue):
                    val = nums[0]
                elif len(nums) >= 1 and re.fullmatch(r"[\s\d.\-+/%()xX×↑↓*†‡]*", c2):
                    val = nums[0]
                else:
                    return {"kind": "text", "text": c, "is_bold": is_bold}
    try:
        v = float(val)
    except ValueError:
        return {"kind": "text", "text": clean_cell(s), "is_bold": is_bold}
    return {"kind": "number", "value": v, "printed": val, "std": (float(std) if std else None),
            "std_printed": std, "is_bold": is_bold, "is_percent": is_percent}


def decimals(printed: str) -> int:
    return len(printed.split(".")[1]) if "." in printed else 0


def rounding_interval(printed: str) -> tuple[float, float]:
    """Interval of true values that print as `printed` under round-half-up at its precision."""
    v = float(printed)
    h = 0.5 * 10 ** (-decimals(printed))
    return (v - h, v + h)


def intervals_overlap(p1: str, p2: str) -> bool:
    a, b = rounding_interval(p1), rounding_interval(p2)
    return a[0] < b[1] and b[0] < a[1]


def _split_rows(body: str) -> list[str]:
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\}|"
                  r"addlinespace(?:\[[^\]]*\])?|specialrule\{[^}]*\}\{[^}]*\}\{[^}]*\}|arrayrulecolor\{[^}]*\}|"
                  r"noalign\{[^}]*\}|rowcolor\{[^}]*\}|morecmidrules)", " ", body)
    # split on \\ only at brace depth 0 (a \\ inside \shortstack{...} or \makecell{...} is a line break in a cell)
    rows, depth, cur, i = [], 0, [], 0
    while i < len(body):
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "\\" and body.startswith("\\\\", i) and depth == 0:
            rows.append("".join(cur)); cur = []
            i += 2
            m = re.match(r"\[[^\]]*\]", body[i:])
            if m:
                i += m.end()
            continue
        cur.append(ch); i += 1
    rows.append("".join(cur))
    return [r for r in rows if "&" in r or r.strip()]


def _split_cells(row: str) -> list[str]:
    cells, depth, cur = [], 0, []
    for ch in row:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    return cells


MULTICOL_RE = re.compile(r"\\multicolumn\s*\{(\d+)\}\s*\{[^{}]*\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
MULTIROW_RE = re.compile(r"\\multirow\s*(?:\[[^\]]*\])?\{(-?\d+|\*)\}\s*(?:\{[^{}]*\}|\*|\[[^\]]*\])?\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


def _expand_row(row: str) -> list[dict]:
    """Expand multicolumn/multirow into a flat cell list with span markers."""
    out = []
    for c in _split_cells(row):
        m = MULTICOL_RE.search(c)
        if m:
            n, content = int(m.group(1)), m.group(2)
            out.append({"raw": content, "span": n, "spanned": False})
            out.extend({"raw": "", "span": 0, "spanned": True} for _ in range(n - 1))
        else:
            mr = MULTIROW_RE.search(c)
            if mr:
                out.append({"raw": mr.group(2), "span": 1, "spanned": False, "multirow": mr.group(1)})
            else:
                out.append({"raw": c, "span": 1, "spanned": False})
    return out


def _is_header_row(cells: list[dict]) -> bool:
    parsed = [parse_number_cell(c["raw"]) for c in cells if c["raw"].strip()]
    nums = sum(1 for p in parsed if p["kind"] == "number")
    return len(parsed) > 0 and nums <= max(0, len(parsed) // 4)


def parse_tables(tex: str, doc_start: int = 0, doc_end=None) -> list[dict]:
    """Parse every tabular in `tex`. Returns a list of table dicts:
    {"label", "caption", "start", "end", "header": [col names], "rows": [{"name", "group",
    "cells": [parsed cells]}], "direction": {col_idx: +1/-1/None}}"""
    doc_end = doc_end if doc_end is not None else len(tex)
    tables = []
    envs = [(m.start(), m.end(), m.group(2)) for m in TABLE_ENV_RE.finditer(tex)]

    def env_for(pos):
        for a, b, body in envs:
            if a <= pos < b:
                return body, a
        return None, None

    for tm in TABULAR_RE.finditer(tex):
        if not (doc_start <= tm.start() < doc_end):
            continue
        body = tm.group(3)
        env_body, env_start = env_for(tm.start())
        cap_src = env_body if env_body is not None else tex[max(0, tm.start() - 1500):tm.end() + 500]
        cap = CAPTION_RE.search(cap_src)
        lab = LABEL_RE.search(cap_src)
        caption = clean_cell(cap.group(1)) if cap else ""
        label = lab.group(1).strip() if lab else None
        rows_raw = [_expand_row(r) for r in _split_rows(body) if r.strip()]
        rows_raw = [r for r in rows_raw if any(c["raw"].strip() for c in r)]
        if len(rows_raw) < 2:
            continue
        header_rows = []
        i = 0
        while i < len(rows_raw) and i < 3 and _is_header_row(rows_raw[i]):
            header_rows.append(rows_raw[i]); i += 1
        if not header_rows:
            header_rows, i = [rows_raw[0]], 1
        ncol = max(len(r) for r in rows_raw)
        header = []
        for j in range(ncol):
            parts = []
            for hr in header_rows:
                if j < len(hr):
                    k = j
                    while k >= 0 and hr[k]["spanned"]:
                        k -= 1
                    txt = clean_cell(hr[k]["raw"]) if k >= 0 else ""
                    if txt and txt not in parts:
                        parts.append(txt)
            header.append(" | ".join(parts))
        rows, group = [], None
        for r in rows_raw[i:]:
            cells = []
            for j in range(ncol):
                raw = r[j]["raw"] if j < len(r) else ""
                cells.append(parse_number_cell(raw) if raw.strip() else {"kind": "empty"})
            first = cells[0]
            if first.get("kind") == "text" and r and r[0].get("multirow"):
                group = first["text"]
            name = first["text"] if first.get("kind") == "text" else (str(first.get("printed")) if first.get("kind") == "number" else "")
            name_eff = group if (first.get("kind") == "empty" and group) else name
            nonempty = [c for c in cells[1:] if c.get("kind") != "empty"]
            if not nonempty and name and r[0].get("span", 1) > 1:
                group = name; continue
            if nonempty and all(c.get("kind") == "text" for c in nonempty) and len(nonempty) <= 1 and r[0].get("span", 1) > 1:
                group = clean_cell(r[0]["raw"]); continue
            rows.append({"name": name_eff, "own_name": name, "group": group if name_eff != group else None,
                         "cells": cells})
        direction = {}
        cl = caption.lower()
        for j, h in enumerate(header):
            hl = h.lower()
            if "↓" in h or "downarrow" in hl or "(lower" in hl or "lower is better" in hl:
                direction[j] = -1
            elif "↑" in h or "uparrow" in hl or "(higher" in hl or "higher is better" in hl:
                direction[j] = +1
            elif LOWER_BETTER.search(hl) and not HIGHER_BETTER.search(hl):
                direction[j] = -1
            elif HIGHER_BETTER.search(hl) and not LOWER_BETTER.search(hl):
                direction[j] = +1
            elif ("lower is better" in cl or "downarrow" in cl) and "higher is better" not in cl:
                direction[j] = -1
            elif ("higher is better" in cl or "uparrow" in cl) and "lower is better" not in cl:
                direction[j] = +1
            else:
                direction[j] = None
        tables.append({"label": label, "caption": caption, "start": tm.start(), "end": tm.end(),
                       "env_start": env_start, "header": header, "rows": rows, "direction": direction,
                       "n_rows": len(rows), "n_cols": ncol})
    return tables


def numeric_cells(table: dict):
    for ri, r in enumerate(table["rows"]):
        for ci, c in enumerate(r["cells"]):
            if c.get("kind") == "number":
                yield ri, ci, c


def find_cells_by_value(tables: list[dict], printed: str) -> list[tuple]:
    """All (table_idx, row_idx, col_idx) whose printed value is rounding-compatible with `printed`."""
    hits = []
    for ti, t in enumerate(tables):
        for ri, ci, c in numeric_cells(t):
            if intervals_overlap(c["printed"], printed):
                hits.append((ti, ri, ci))
    return hits
