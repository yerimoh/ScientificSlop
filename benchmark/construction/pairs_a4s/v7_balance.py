"""Balance table and alignment for the v7 A4S pairs, with the same TeX reader on both sides.

This is the table the earlier PDF-versus-TeX comparison could not produce. Every count below is
read by slopbench_lib.metrics from LaTeX on both sides, so a gap is a property of the papers and
not of the reader. SPECTER2 counterpart rank mirrors the paper's alignment table.
"""
import json, os, statistics as st

OUTD = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/data/Agents4Science"
FEATS = ["body_words", "n_sections", "n_subsections", "n_figures", "n_tables", "n_equations",
         "n_unique_cites", "n_internal_refs", "appendix_words"]


def auroc(pos, neg):
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    n = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return round(n / (len(pos) * len(neg)), 3)


def specter_rank(P):
    """Rank of each pair's true counterpart among all human papers, by SPECTER2 cosine."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        return None, f"transformers unavailable ({type(e).__name__})"
    name = "allenai/specter2_base"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name).eval()
    except Exception as e:
        return None, f"model load failed ({type(e).__name__})"

    def emb(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                b = tok(texts[i:i + 8], padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
                out.append(mod(**b).last_hidden_state[:, 0, :])
        x = torch.cat(out)
        return x / x.norm(dim=1, keepdim=True)

    sep = tok.sep_token or " "
    A = emb([p["ai"]["title"] + sep + (p["ai"]["abstract"] or "")[:1500] for p in P])
    H = emb([p["human"]["title"] + sep + (p["human"]["abstract"] or "")[:1500] for p in P])
    S = A @ H.T
    ranks = [int((S[i] > S[i, i]).sum().item()) + 1 for i in range(len(P))]
    diag = [round(float(S[i, i]), 3) for i in range(len(P))]
    return (ranks, diag), None


def main():
    d = json.load(open(f"{OUTD}/pairs_a4s_v7.json"))
    P = d["pairs"]
    L = [f"# Balance and alignment, Agents4Science v7 pairs", "",
         f"{len(P)} pairs. Both sides read from LaTeX with the same parser, so these counts are "
         f"comparable in a way the PDF-based table for the 118-pair set was not.", "",
         "| feature | AI median | human median | ratio | AUROC (AI vs human) | separability |",
         "|---|---|---|---|---|---|"]
    for f in FEATS:
        ai = [p["ai"].get(f) for p in P]
        hu = [p["human"].get(f) for p in P]
        a, h = st.median(ai), st.median(hu)
        au = auroc(ai, hu)
        L.append(f"| {f} | {a} | {h} | {round(a / h, 2) if h else None} | {au} | "
                 f"{round(max(au, 1 - au), 3) if au is not None else None} |")
    r = [p["body_ratio"] for p in P if p["body_ratio"]]
    L += ["", f"Body-word ratio (human/AI) median {st.median(r):.2f}, min {min(r):.2f}, "
              f"max {max(r):.2f}; {sum(1 for x in r if x > 2.5)} pairs exceed 2.5.",
          f"Appendices are present in {sum(1 for p in P if p['ai']['has_appendix'])} AI papers and "
          f"{sum(1 for p in P if p['human']['has_appendix'])} human papers; "
          f"{sum(1 for p in P if p['human']['github'])} human and "
          f"{sum(1 for p in P if p['ai']['github'])} AI papers link a repository. "
          f"A related-work section is present in {sum(1 for p in P if p['ai']['has_related_work'])} AI "
          f"and {sum(1 for p in P if p['human']['has_related_work'])} human papers.", ""]
    got, err = specter_rank(P)
    if got:
        ranks, diag = got
        L += [f"SPECTER2 counterpart rank among the {len(P)} human papers: median {st.median(ranks)}, "
              f"rank 1 for {sum(1 for x in ranks if x == 1)} pairs, worst {max(ranks)}. "
              f"Median pair cosine {st.median(diag):.3f}.", ""]
        for p, rk, cs in zip(P, ranks, diag):
            p["specter2_rank"] = rk
            p["specter2_cos"] = cs
        json.dump(d, open(f"{OUTD}/pairs_a4s_v7.json", "w"), indent=1, ensure_ascii=False)
    else:
        L += [f"SPECTER2 alignment not computed ({err}).", ""]
    open(f"{OUTD}/BALANCE_a4s_v7.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
