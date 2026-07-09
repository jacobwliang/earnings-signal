# Fine-tuned Correlation Analysis (ES-12)

## Purpose

Test whether the fine-tuned FinBERT's call-level sentiment correlates with the
**company-specific** part of the post-earnings move, and whether it beats the
baseline chance floor. The primary endpoint is the **1-day abnormal return**
(`abn_return_1d = return_1d − market_return_1d`, SPY as the market, beta = 1,
same business-day window as the raw return) — raw returns are dominated by
market-wide moves, so subtracting the market isolates what sentiment could
plausibly predict. See [baseline_analysis.md](baseline_analysis.md) for the
scores and the chance-floor construction.

## What was run

- **Unit of analysis:** one row per call (13,611 calls). The CEO and CFO
  rows of a call share its return, so their sentiment is averaged to the call
  level; keeping both would be pseudo-replication. Single-speaker calls (~12%)
  fall out of the same mean with no special-casing.
- **Statistic:** Spearman rho (rank-based, robust to the return tails).
- **Inference:** a **ticker-clustered** bootstrap (10,000 resamples, seed
  42) — a ticker's calls are not independent, so whole tickers are resampled
  together. Significance is judged by whether the 95% CI excludes zero. The
  `p_value` column is scipy's asymptotic Spearman p as a cross-check only; it
  ignores clustering and is anti-conservative.
- **Multiple testing:** Benjamini-Hochberg across the fine-tuned robustness grid
  (raw/abnormal × 1d/5d). The baseline floor and the difference test answer
  different questions and are reported outside that family.

## Results

**Primary — fine-tuned sentiment vs `abn_return_1d`:** rho = +0.1035,
95% CI [+0.0866, +0.1200] (n = 13,429).
The CI **excludes zero**.

**Baseline floor (same endpoint):** rho = -0.0024,
95% CI [-0.0192, +0.0148]. Expected to sit at
zero — the baseline head is untrained.

**Fine-tuned − baseline (paired):** Δrho = +0.1059,
95% CI [+0.0821, +0.1290]. The CI **does**
exclude zero (bootstrap p = <0.001).

**Minimum detectable effect:** with n = 13,429 calls the i.i.d. floor
is ≈ ±0.009; ticker clustering widens it. Read small non-significant rhos as
*underpowered / near-zero*, not as proof of no relationship.

### Robustness across return windows (BH-corrected)

| return_col | n | rho | 95% CI | p (asymp.) | p (BH) | CI excl. 0 |
|---|---|---|---|---|---|---|
| `return_1d` | 13429 | +0.1035 | [+0.0867, +0.1198] | <0.001 | <0.001 | yes |
| `abn_return_1d` | 13429 | +0.1035 | [+0.0866, +0.1200] | <0.001 | <0.001 | yes |
| `return_5d` | 13507 | +0.0829 | [+0.0662, +0.1000] | <0.001 | <0.001 | yes |
| `abn_return_5d` | 13507 | +0.0896 | [+0.0729, +0.1065] | <0.001 | <0.001 | yes |

## Known limitations

- **Abnormal return is beta = 1 (market-excess), not a fitted market model** — no
  per-ticker beta or estimation window. This removes market-wide moves but not
  systematic beta exposure; it is the low-lookahead choice.
- **Asymptotic p-values ignore clustering** — the clustered bootstrap CI is the
  trustworthy inference; the p columns are cross-checks.
- **Speaker-source and subgroup cuts (ES-13) are out of scope here** — this is
  the call-level primary analysis only.
- **Missing returns** — `return_1d`/`return_5d` carry a few hundred nulls;
  each correlation drops rows null in its own pair.

## Reproduce

```bash
.venv/bin/python -m src.data.download_index
.venv/bin/python -m src.analysis.correlate_returns --n-boot 10000 --seed 42
```

## Outputs

- `results/correlation_results.csv` / `.json` — tidy, one row per analysis piece.
- `reports/finetuned_correlation_analysis.md` — this report.
