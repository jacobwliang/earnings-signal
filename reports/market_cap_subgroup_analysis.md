# Market-Cap Subgroup Analysis (ES-13)

## Purpose

Does the fine-tuned sentiment → post-earnings-return signal
([finetuned_correlation_analysis.md](finetuned_correlation_analysis.md), pooled rho ≈
+0.10) depend on **company size**? Small caps are less-covered and slower to price news,
so a size-conditional signal is the natural place to look for a stronger — or a
spurious — edge. This cuts the same call-level correlation into market-cap terciles.

## What was run

- **Unit:** one row per call (same aggregation as the primary analysis), scored by the
  fine-tuned model, against the 1-day **abnormal** return (`return_1d − market_return_1d`).
- **Size buckets:** market cap **as of the earnings date** (shares × unadjusted close at
  that date), split into small/mid/large **terciles within each earnings-date quarter** —
  cross-sectional ranking so a market-wide drift in caps can't push later quarters into a
  different bucket.
- **Statistic:** per-tercile Spearman rho with a **ticker-clustered** bootstrap 95% CI
  (10,000 resamples, seed 42) — the exact `spearman_ci` used in the primary analysis, so
  clustering is handled, not just caveated.
- **Difference test:** two terciles are read as "differing" only when their 95% CIs
  **don't overlap** — a plain, conservative pairwise read, no p-value.

## Results

**Primary — fine-tuned sentiment vs `abn_return_1d`, by size tercile:**

| tercile | n | rho | 95% CI | CI excl. 0 |
|---|---|---|---|---|
| small | 4,455 | +0.121 | [+0.090, +0.151] | yes |
| mid | 4,442 | +0.101 | [+0.074, +0.127] | yes |
| large | 4,428 | +0.081 | [+0.052, +0.110] | yes |

**Pairwise:** all three CIs overlap → **no pair is statistically distinguishable**
(small-mid, small-large, mid-large all "no difference").

**Verdict:** the signal is **present and significant in every size bucket** — it is not
concentrated in small caps or driven by them. There is a clean monotone gradient
(small > mid > large) hinting the edge is stronger for smaller companies, but the CIs
overlap, so that gradient is **suggestive, not established**.

**Robustness:** the ordering and the "all overlap" verdict hold across all four windows
(`return_1d`, `abn_return_5d`, `return_5d`); small stays highest (rho ≈ 0.09–0.12),
large lowest (≈ 0.07–0.08) in every one.

## Known limitations

- **Underpowered to separate terciles.** Each bucket is ~4,400 calls (a third of the
  data) and CIs are clustered/wide, so overlapping CIs mean *we can't resolve a
  difference*, **not** that the terciles are equal. The small>large gradient is the
  kind of effect this design cannot confirm or rule out.
- **CI-overlap is conservative** — non-overlap implies a difference, but overlap does not
  imply equality (the honest reading of the row above).
- **105 calls (of 13,611) lack a market cap** (shares unavailable at the earnings date)
  and are dropped; ~180–290 more drop per window for a missing return.
- **Beta = 1 abnormal return**, inherited from the primary analysis — market-excess, not
  a fitted market model.

## Reproduce

```bash
# from repo root; writes results/subgroup_market_cap_results.json
.venv/bin/python -m src.analysis.subgroup_market_cap
```

## Outputs

- `results/subgroup_market_cap_results.json` — per-tercile rho/CI and the
  pairwise-overlap rows, all four return windows.
- `reports/market_cap_subgroup_analysis.md` — this report.
