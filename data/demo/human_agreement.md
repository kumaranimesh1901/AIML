<!-- human-check:start -->
## Judge vs Real Human Annotation

Source: **real human annotation** (two people labelled each answer pass or fail without seeing the judge's decision). This replaces the simulated-annotator agreement in the judge validation section, whose annotators copy the judge.

- **Judge**: `qwen2.5:14b`, version `j1.5`
- **Items in sheet**: 50; filled by annotator 1: 36, annotator 2: 36
- **Fail rate on these items**: judge 75.0%, annotator 1 33.3%, annotator 2 88.9%

| Comparison | n | Cohen's kappa | 95% CI (bootstrap) | Raw agreement | Reading |
|---|---|---|---|---|---|
| Judge vs human consensus (items both annotators agree on) | 16 | -0.2727 | -0.47 to 0.00 | 56.2% | poor |
| Judge vs annotator 1 | 36 | 0.0000 | -0.23 to 0.23 | 41.7% | poor |
| Judge vs annotator 2 | 36 | -0.1818 | -0.30 to -0.05 | 63.9% | poor |
| Annotator 1 vs annotator 2 (human ceiling) | 36 | 0.1176 | 0.02 to 0.26 | 44.4% | slight |

Against the human consensus: both fail 9, judge fail but humans pass 4, judge pass but humans fail 3, both pass 0. Judge fail precision 69.2%, recall 75.0%. The annotators disagreed on 20 items, which are left out of the consensus row.

_Generated 2026-11-23T12:48:38+00:00 by `edg calibrate human-import`._
<!-- human-check:end -->
