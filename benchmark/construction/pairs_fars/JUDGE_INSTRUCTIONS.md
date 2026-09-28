# Tier-judging instructions (given to each judging agent, 2026-09-11)

Input: one batch file `cache/judge_batches/batch_XX.json` — a list of FARS papers, each with
`code`, `fars_title`, `fars_abstract`, `fars_heur_type`, and `candidates[]`
(`arxiv`, `title`, `year`, `venue`, `relation`, `abstract`).

Output: write `cache/judge_out/batch_XX.json` with, for every FARS code:

```json
{"code": "FA0001",
 "fars_type": "method|framework|benchmark|dataset|analysis|survey",
 "fars_type_reason": "one line",
 "candidates": {"2405.16833": {"type": "method", "tier": 3, "reason": "one line"}}}
```

## Paper type (PAIR_RULE_0911.md step 0)
- method    : proposes a new method/algorithm/pipeline and has a method section
- framework : proposes an evaluation/analysis framework or metric system
- benchmark : builds a task/data/leaderboard with a data-construction section
- dataset   : the data itself is the contribution
- analysis  : measures/dissects a phenomenon without proposing a new method
  (negative-result papers "X does not improve Y" that mainly TEST a hypothesis are analysis;
   if the paper's core is a new proposed technique that happens to fail, still method)
- survey    : organizes prior work

## Similarity tier (PAIR_RULE_0911.md step 3, judged on title+abstract)
- 3 : same problem + same kind of contribution
- 2 : same problem, different kind of contribution
- 1 : same area, different problem
- 0 : cited only as a tool/model/dataset (e.g. a base-model tech report, a benchmark the FARS
      paper merely evaluates on)

Judge every candidate listed. Keep reasons to one short line each. Do not invent candidates,
do not drop any. Base the judgment only on the provided titles and abstracts.
