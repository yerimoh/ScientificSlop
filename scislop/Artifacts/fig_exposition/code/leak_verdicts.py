"""Hand verdicts for the experimental-content kind of `fig_exposition` (the former item G,
`fig_leak`). This kind is the one element the deterministic patterns cannot decide, because whether a
number is a run setting or a constant of the method is a reading of what the number means.

The rule. A number is EXPERIMENTAL CONTENT when it names something the experimenter chose for the run
(optimiser, learning rate, epochs or steps, batch size, seeds, GPUs or GPU-hours, the size of the
training or evaluation set, the context length of the evaluation inputs) or a value the run produced
(a rate, a share, a reduction, a speed-up, a score), or when a panel of the figure lists metrics or
benchmarks as a stage of the method. It is a METHOD CONSTANT, and therefore borderline and not counted,
when it belongs to the method's own definition (number of drafts or candidates, cluster or ensemble
size, diffusion steps, thresholds and acceptance ranges, the input tensor shape, the size of a probe or
canary set the method draws its signal from, and token counts that parameterise the method such as a
prefix length, a retrieval budget or a representation size). Optimiser names are method components in
an optimiser paper. Values of a drawn worked example, axis labels of a result plot embedded in the
figure, model names and caption text are not counted.

Every entry was read in its transcript line with the lines around it, and every figure was opened. One
reader; a second reader has to repeat the verdicts before a rate is quoted. Provenance of the verdicts
and of the false-negative sweep that added to them is
`../../candidte/_census_0914/{pairs165_verdicts.py, scale_verdicts.py}`.
"""

# Counted. FARS method figures of the benchmark pairs.
CLEAR_AI = {
    "FA0002": "Context Length: 4K tokens (run configuration)",
    "FA0006": "Triggered only on disagreement (~10% of cases) (measured share)",
    "FA0012": "~30% of objects (measured share)",
    "FA0015": "Evaluation panel",
    "FA0018": "~60% / ~27% zero-variance groups (measured)",
    "FA0020": "Compute: ~5 min CPU vs 16 GPU-hours (measured cost; broad config rescan 0915)",
    "FA0022": "51,200 samples x 1 epoch, 1,600 x 32 epochs, warmup; ParseRate ~43/84/49%",
    "FA0031": "1 call ~706 tokens, 3 calls ~1695 / ~1998 tokens (measured cost)",
    "FA0034": "Easy queries (63%) / Medium (32%) / Hard (5%) (measured shares)",
    "FA0040": "~85% / ~10% Failures, ~90% Success",
    "FA0055": "~90% unchanged parameters (measured)",
    "FA0059": "262K tokens, 4096 tokens (context configuration)",
    "FA0074": "99.78% fallback rate (measured)",
    "FA0101": "Latency reduced by ~59%",
    "FA0110": "Evaluation Flow panel",
    "FA0112": "33% Secure Code attacked / 99% Secure Code with noisy quantisation (measured)",
    "FA0114": "1.82x fewer judge prompts, 0% accuracy loss (measured)",
    "FA0123": "x 1 epoch, x 32 epochs, 1.6k samples",
    "FA0147": "Evaluation panel (twice)",
    "FA0156": "1 epoch / 32 epochs; Acc@k 25.6 / 25.5 / 38.3% (best); Condition A: 25.6%",
    "FA0161": "99.98% pass",
    "FA0162": "Recall: 51.3 / 84.8 / 72.6%",
    "FA0168": "8,632 steps/epoch vs 64 steps/epoch, Step reduction: 135x (training setting and result; broad config rescan 0915)",
    "FA0181": "Only ~14% of channels (measured)",
    "FA0201": "87.9% padding, 99% of counts < C, only 5.4% overflow (measured)",
    "FA0208": "c->i 36.25% (High Regression) / 2.76% (Low Regression) (measured)",
    "FA0209": "Adam, lr=5e-4, 25 epochs; 3000 edits; Evaluation panel; Capability (MMLU, NQ, SST-2, WMT, GSM8K)",
    "FA0214": "Truncated Description (40 tokens), N=100; (74%), 11x welfare improvement",
    "FA0218": "~50% KV reduction",
    "FA0221": "(N=60) sample count",
    "FA0237": "Pc=52.4% / 32.5% / 74.8% (measured)",
    "FA0280": "Accuracy 55% / 39% / 69% / 66% (measured)",
    "FA0292": "LR = 5e-5 on three boxes, LR = 1e-5; three Evaluation boxes",
}

# Counted. Human method figures of the pair partners.
CLEAR_HU = {
    "2403.06764": "method figure annotates each variant with its measured cost (100% / 52% / 38% / 33% FLOPs)",
    "2504.02010": "pipeline figure carries a Tasks panel (AIME 2024, FOLIO, MuSiQue) and Verification of Findings",
}

# Not counted. Method constants and other readings that do not settle.
BORDER_AI = {
    "FA0001": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0013": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0021": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0023": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0028": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0035": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0058": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0067": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0072": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0075": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0085": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0115": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0163": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0172": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0184": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0186": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0188": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0193": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0194": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0205": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0255": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0309": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0336": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0374": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0388": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
}

# Not counted. Detector nominations read and rejected.
FALSE_AI = {
    "FA0038": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0044": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0069": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0137": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0208": "c->i 36.25% (High Regression) / 2.76% (Low Regression) (measured)",
    "FA0234": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "FA0349": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
}

# Not counted.
BORDER_HU = {
    "2310.17631": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "2401.10480": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
}

# Not counted.
FALSE_HU = {
    "2305.16264": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "2310.02575": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
    "2402.04333": "hand-verified clear (see _census_0914/pairs165_verdicts.py)",
}

CLEAR = {**CLEAR_AI, **CLEAR_HU}
