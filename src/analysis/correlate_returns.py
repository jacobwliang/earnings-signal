"""Correlate document sentiment with forward returns, fine-tuned vs baseline.

The headline question is whether the fine-tuned FinBERT's call-level sentiment
relates to the *company-specific* part of the post-earnings move. Raw 1d/5d
returns are dominated by broad market moves, so the primary endpoint is a
**market-adjusted abnormal return** (``stock_return - market_return``, beta = 1,
same business-day window); raw returns are kept as a robustness comparison.

Inference layer:

- ``add_abnormal_returns`` merges a per-date market (SPY) return onto the scores
  and subtracts it. The market return is date-driven and ticker-independent, so
  it is derived here at analysis time — no pipeline re-run — via
  ``compute_market_returns`` in :mod:`src.data.compute_returns`.
- ``aggregate_to_call`` collapses the (CEO, CFO) speaker rows to one row per call
  (both share the call's return; keeping both would be pseudo-replication).
- ``spearman_ci`` is the core statistic: a point Spearman rho with a
  **ticker-clustered** bootstrap CI (a ticker's calls are not independent).
  Significance is judged by whether that CI excludes zero; the reported
  ``p_value`` is scipy's asymptotic Spearman p as a *labelled cross-check* only
  (it ignores clustering, so it is anti-conservative).
- ``bootstrap_difference`` tests fine-tuned minus baseline on the *same* calls,
  drawing one ticker resample per iteration and evaluating both models on it.
- ``robustness_table`` runs the grid of return windows and applies
  Benjamini-Hochberg across that family.

``main`` writes a tidy ``correlation_results.csv`` (+ JSON) and renders
``reports/finetuned_correlation_analysis.md``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.compute_returns import compute_market_returns

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCORES_PATH = ROOT / "results" / "baseline_scores.parquet"
FINETUNED_SCORES_PATH = ROOT / "results" / "finetuned_scores.parquet"
INDEX_PATH = ROOT / "data" / "raw" / "index_prices.parquet"
OUTPUT_CSV_PATH = ROOT / "results" / "correlation_results.csv"
OUTPUT_JSON_PATH = ROOT / "results" / "correlation_results.json"
OUTPUT_REPORT_PATH = ROOT / "reports" / "finetuned_correlation_analysis.md"

DEFAULT_N_BOOT = 10000
DEFAULT_SEED = 42

PRIMARY_RETURN_COL = "abn_return_1d"
# The multiple-testing family corrected together (BH) — the fine-tuned model
# across raw/abnormal x 1d/5d. The baseline floor and the difference test are
# reported separately (they answer different questions) and excluded from it.
ROBUSTNESS_RETURN_COLS = ("return_1d", "abn_return_1d", "return_5d", "abn_return_5d")

# Per-call invariants carried through aggregation unchanged (identical within a
# call, so ``first`` is exact). The four return columns are model-independent.
_CALL_CARRY_COLS = ("ticker", "return_start_date", "return_1d", "return_5d",
                    "abn_return_1d", "abn_return_5d")


# --------------------------------------------------------------------------- #
# Data shaping
# --------------------------------------------------------------------------- #
def add_abnormal_returns(scores_df: pd.DataFrame, market_returns: pd.DataFrame) -> pd.DataFrame:
    """Add ``abn_return_1d`` / ``abn_return_5d`` = raw return minus market return.

    Merges ``market_returns`` (one row per date, from ``compute_market_returns``)
    onto the scores by ``return_start_date`` and subtracts. Both sides are
    normalized to midnight Timestamps first so the date keys align regardless of
    whether the scores stored ``datetime.date`` objects.
    """
    df = scores_df.copy()
    df["return_start_date"] = pd.to_datetime(df["return_start_date"]).dt.normalize()

    mr = market_returns.copy()
    mr["return_start_date"] = pd.to_datetime(mr["return_start_date"]).dt.normalize()

    merged = df.merge(mr, on="return_start_date", how="left")
    merged["abn_return_1d"] = merged["return_1d"] - merged["market_return_1d"]
    merged["abn_return_5d"] = merged["return_5d"] - merged["market_return_5d"]
    return merged


def aggregate_to_call(df: pd.DataFrame, weight_by_chunks: bool = False) -> pd.DataFrame:
    """Collapse the per-speaker rows to one row per ``transcript_id`` (a call).

    ``sentiment_score`` is the plain mean across the call's speaker rows (or an
    ``n_chunks``-weighted mean when ``weight_by_chunks`` is set). This handles the
    single-speaker calls (~12% of the data) with no special-casing — a mean over
    one row is that row. ``n_chunks`` is summed; ticker/date and all four return
    columns are carried via ``first`` (identical across a call's speaker rows).
    """
    grouped = df.groupby("transcript_id", sort=True)

    if weight_by_chunks:
        weighted = df.assign(
            _num=df["sentiment_score"] * df["n_chunks"], _den=df["n_chunks"]
        ).groupby("transcript_id", sort=True)[["_num", "_den"]].sum()
        sentiment = (weighted["_num"] / weighted["_den"]).rename("sentiment_score")
    else:
        sentiment = grouped["sentiment_score"].mean()

    agg = grouped.agg(
        n_chunks=("n_chunks", "sum"),
        **{col: (col, "first") for col in _CALL_CARRY_COLS},
    )
    out = agg.join(sentiment).reset_index()
    return out[["transcript_id", "sentiment_score", "n_chunks", *_CALL_CARRY_COLS]]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho as Pearson correlation of average ranks (tie-aware).

    Equivalent to ``scipy.stats.spearmanr(...).statistic`` but without the
    p-value machinery, so it is cheap enough to call inside the bootstrap loop.
    Returns NaN when either variable is constant on the sample.
    """
    xr = stats.rankdata(x)
    yr = stats.rankdata(y)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom == 0:
        return float("nan")
    return float((xr * yr).sum() / denom)


def _ticker_index_groups(tickers: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (unique_tickers, list of positional-index arrays per ticker)."""
    order = np.argsort(tickers, kind="stable")
    sorted_t = tickers[order]
    unique, starts = np.unique(sorted_t, return_index=True)
    groups = np.split(order, starts[1:])
    return unique, groups


def _cluster_bootstrap(
    x: np.ndarray, y: np.ndarray, tickers: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """Bootstrap Spearman rho by resampling whole tickers with replacement.

    Each draw picks ``k`` tickers (k = number of clusters) with replacement and
    pools all their rows, so within-ticker dependence is preserved. Returns the
    array of ``n_boot`` resampled rhos.
    """
    _, groups = _ticker_index_groups(tickers)
    k = len(groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, k, size=k)
        idx = np.concatenate([groups[d] for d in drawn])
        boot[b] = _spearman_rho(x[idx], y[idx])
    return boot


def _ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    """True if the (low, high) interval lies entirely on one side of zero."""
    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)


def spearman_ci(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    ticker_col: str = "ticker",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci_level: float = 0.95,
) -> dict:
    """Point Spearman rho with a ticker-clustered bootstrap percentile CI.

    Drops rows null in either variable, computes the point rho, then bootstraps
    over tickers for the CI. ``p_value`` is scipy's asymptotic two-sided Spearman
    p-value — a cross-check only; it ignores clustering, so treat the CI as the
    real inference. ``boot_rhos`` is the raw resample array.
    """
    pair = df[[score_col, return_col, ticker_col]].dropna(subset=[score_col, return_col])
    x = pair[score_col].to_numpy(dtype=float)
    y = pair[return_col].to_numpy(dtype=float)
    tickers = pair[ticker_col].to_numpy()

    result = stats.spearmanr(x, y)
    boot_rhos = _cluster_bootstrap(x, y, tickers, n_boot, seed)
    alpha = 1.0 - ci_level
    ci_low = float(np.nanpercentile(boot_rhos, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(boot_rhos, 100 * (1 - alpha / 2)))

    return {
        "n": int(len(pair)),
        "rho": float(result.statistic),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(result.pvalue),
        "significant": _ci_excludes_zero(ci_low, ci_high),
        "boot_rhos": boot_rhos,
    }


def bootstrap_difference(
    finetuned_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    score_col: str,
    return_col: str,
    ticker_col: str = "ticker",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci_level: float = 0.95,
) -> dict:
    """Paired bootstrap of ``rho_finetuned - rho_baseline`` on the same calls.

    The two frames are the same calls scored by different models, so the test is
    paired: each iteration draws one ticker resample and evaluates *both* models
    on it. Asserts the frames share the same ``transcript_id`` set before pairing.
    The CI is the primary inference; ``p_value`` is a percentile bootstrap
    two-sided achieved-significance level (no analytic null exists for a
    difference of dependent Spearman correlations).
    """
    ft_ids = set(finetuned_df["transcript_id"])
    bl_ids = set(baseline_df["transcript_id"])
    assert ft_ids == bl_ids, "finetuned and baseline must cover the same transcript_ids"

    ft = finetuned_df.set_index("transcript_id")
    bl = baseline_df.set_index("transcript_id")
    data = pd.DataFrame({
        "ticker": ft[ticker_col],
        "score_ft": ft[score_col],
        "score_bl": bl.loc[ft.index, score_col].to_numpy(),
        "ret": ft[return_col],
    }).dropna(subset=["score_ft", "score_bl", "ret"])

    xf = data["score_ft"].to_numpy(dtype=float)
    xb = data["score_bl"].to_numpy(dtype=float)
    y = data["ret"].to_numpy(dtype=float)
    tickers = data["ticker"].to_numpy()

    diff = _spearman_rho(xf, y) - _spearman_rho(xb, y)

    _, groups = _ticker_index_groups(tickers)
    k = len(groups)
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, k, size=k)
        idx = np.concatenate([groups[d] for d in drawn])
        boot_diffs[b] = _spearman_rho(xf[idx], y[idx]) - _spearman_rho(xb[idx], y[idx])

    alpha = 1.0 - ci_level
    ci_low = float(np.nanpercentile(boot_diffs, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(boot_diffs, 100 * (1 - alpha / 2)))
    prop_le = float(np.mean(boot_diffs <= 0))
    prop_ge = float(np.mean(boot_diffs >= 0))
    p_value = float(min(1.0, 2 * min(prop_le, prop_ge)))

    return {
        "n": int(len(data)),
        "diff": float(diff),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": _ci_excludes_zero(ci_low, ci_high),
        "boot_diffs": boot_diffs,
    }


def robustness_table(
    df: pd.DataFrame,
    score_col: str = "sentiment_score",
    return_cols: tuple[str, ...] = ROBUSTNESS_RETURN_COLS,
    ticker_col: str = "ticker",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Spearman CI per return window, with Benjamini-Hochberg across the family.

    Runs :func:`spearman_ci` once per column and applies BH (``scipy.stats.
    false_discovery_control``) across just these p-values — the correction family
    is exactly this grid.
    """
    rows = []
    for col in return_cols:
        r = spearman_ci(df, score_col, col, ticker_col, n_boot, seed)
        rows.append({
            "return_col": col,
            "n": r["n"],
            "rho": r["rho"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p_value": r["p_value"],
            "significant": r["significant"],
        })
    table = pd.DataFrame(rows)
    table["p_adjusted"] = stats.false_discovery_control(table["p_value"].to_numpy(), method="bh")
    return table


# --------------------------------------------------------------------------- #
# Orchestration & output
# --------------------------------------------------------------------------- #
def _min_detectable_rho(n: int) -> float:
    """Rough i.i.d. minimum detectable Spearman rho (~1/sqrt(n)); clustering widens it."""
    return float("nan") if n <= 0 else 1.0 / np.sqrt(n)


def assemble_results(primary, baseline, diff, table, n_boot, seed) -> pd.DataFrame:
    """Build the tidy one-row-per-analysis-piece results frame."""
    records = [
        {"analysis": "primary", "model": "finetuned", "return_col": PRIMARY_RETURN_COL,
         "n": primary["n"], "estimate": primary["rho"], "ci_low": primary["ci_low"],
         "ci_high": primary["ci_high"], "p_value": primary["p_value"],
         "p_adjusted": np.nan, "significant": primary["significant"]},
        {"analysis": "baseline_floor", "model": "baseline", "return_col": PRIMARY_RETURN_COL,
         "n": baseline["n"], "estimate": baseline["rho"], "ci_low": baseline["ci_low"],
         "ci_high": baseline["ci_high"], "p_value": baseline["p_value"],
         "p_adjusted": np.nan, "significant": baseline["significant"]},
        {"analysis": "difference", "model": "finetuned-baseline", "return_col": PRIMARY_RETURN_COL,
         "n": diff["n"], "estimate": diff["diff"], "ci_low": diff["ci_low"],
         "ci_high": diff["ci_high"], "p_value": diff["p_value"],
         "p_adjusted": np.nan, "significant": diff["significant"]},
    ]
    for _, row in table.iterrows():
        records.append({
            "analysis": "robustness", "model": "finetuned", "return_col": row["return_col"],
            "n": int(row["n"]), "estimate": row["rho"], "ci_low": row["ci_low"],
            "ci_high": row["ci_high"], "p_value": row["p_value"],
            "p_adjusted": row["p_adjusted"], "significant": bool(row["significant"]),
        })
    out = pd.DataFrame.from_records(records)
    out["n_boot"] = n_boot
    out["seed"] = seed
    return out


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def render_report(results: pd.DataFrame, primary, baseline, diff, table,
                  n_calls: int, n_boot: int, seed: int) -> str:
    """Render the narrative markdown report (baseline_analysis.md template)."""
    mde = _min_detectable_rho(primary["n"])
    verdict = "excludes zero" if primary["significant"] else "includes zero"
    diff_verdict = "does" if diff["significant"] else "does not"

    rob_rows = "\n".join(
        f"| `{r.return_col}` | {int(r.n)} | {r.rho:+.4f} | {_fmt_ci(r.ci_low, r.ci_high)} "
        f"| {_fmt_p(r.p_value)} | {_fmt_p(r.p_adjusted)} | {'yes' if r.significant else 'no'} |"
        for r in table.itertuples()
    )

    return f"""# Fine-tuned Correlation Analysis (ES-12)

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

- **Unit of analysis:** one row per call ({n_calls:,} calls). The CEO and CFO
  rows of a call share its return, so their sentiment is averaged to the call
  level; keeping both would be pseudo-replication. Single-speaker calls (~12%)
  fall out of the same mean with no special-casing.
- **Statistic:** Spearman rho (rank-based, robust to the return tails).
- **Inference:** a **ticker-clustered** bootstrap ({n_boot:,} resamples, seed
  {seed}) — a ticker's calls are not independent, so whole tickers are resampled
  together. Significance is judged by whether the 95% CI excludes zero. The
  `p_value` column is scipy's asymptotic Spearman p as a cross-check only; it
  ignores clustering and is anti-conservative.
- **Multiple testing:** Benjamini-Hochberg across the fine-tuned robustness grid
  (raw/abnormal × 1d/5d). The baseline floor and the difference test answer
  different questions and are reported outside that family.

## Results

**Primary — fine-tuned sentiment vs `abn_return_1d`:** rho = {primary['rho']:+.4f},
95% CI {_fmt_ci(primary['ci_low'], primary['ci_high'])} (n = {primary['n']:,}).
The CI **{verdict}**.

**Baseline floor (same endpoint):** rho = {baseline['rho']:+.4f},
95% CI {_fmt_ci(baseline['ci_low'], baseline['ci_high'])}. Expected to sit at
zero — the baseline head is untrained.

**Fine-tuned − baseline (paired):** Δrho = {diff['diff']:+.4f},
95% CI {_fmt_ci(diff['ci_low'], diff['ci_high'])}. The CI **{diff_verdict}**
exclude zero (bootstrap p = {_fmt_p(diff['p_value'])}).

**Minimum detectable effect:** with n = {primary['n']:,} calls the i.i.d. floor
is ≈ ±{mde:.3f}; ticker clustering widens it. Read small non-significant rhos as
*underpowered / near-zero*, not as proof of no relationship.

### Robustness across return windows (BH-corrected)

| return_col | n | rho | 95% CI | p (asymp.) | p (BH) | CI excl. 0 |
|---|---|---|---|---|---|---|
{rob_rows}

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
.venv/bin/python -m src.analysis.correlate_returns --n-boot {n_boot} --seed {seed}
```

## Outputs

- `results/correlation_results.csv` / `.json` — tidy, one row per analysis piece.
- `reports/finetuned_correlation_analysis.md` — this report.
"""


def run_analysis(baseline_scores, finetuned_scores, index_close, n_boot, seed):
    """Compute market returns, build abnormal returns, aggregate, and run all tests."""
    dates = pd.concat([baseline_scores["return_start_date"],
                       finetuned_scores["return_start_date"]]).unique()
    market_returns = compute_market_returns(index_close, dates)
    print(f"Computed market returns for {len(market_returns)} unique dates")

    baseline = aggregate_to_call(add_abnormal_returns(baseline_scores, market_returns))
    finetuned = aggregate_to_call(add_abnormal_returns(finetuned_scores, market_returns))
    print(f"Aggregated to {len(finetuned)} calls (finetuned), {len(baseline)} (baseline)")

    primary = spearman_ci(finetuned, "sentiment_score", PRIMARY_RETURN_COL, n_boot=n_boot, seed=seed)
    baseline_primary = spearman_ci(baseline, "sentiment_score", PRIMARY_RETURN_COL, n_boot=n_boot, seed=seed)
    diff = bootstrap_difference(finetuned, baseline, "sentiment_score", PRIMARY_RETURN_COL,
                                n_boot=n_boot, seed=seed)
    table = robustness_table(finetuned, n_boot=n_boot, seed=seed)

    results = assemble_results(primary, baseline_primary, diff, table, n_boot, seed)
    report = render_report(results, primary, baseline_primary, diff, table,
                           len(finetuned), n_boot, seed)
    return results, report


def _print_summary(results: pd.DataFrame) -> None:
    print("\nCorrelation results:")
    cols = ["analysis", "model", "return_col", "n", "estimate", "ci_low", "ci_high", "significant"]
    print(results[cols].to_string(index=False))


def main(argv=None) -> None:
    """Load both scored parquets and the index, run the analysis, write outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_SCORES_PATH)
    parser.add_argument("--finetuned", type=Path, default=FINETUNED_SCORES_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV_PATH)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON_PATH)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT_PATH)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    print(f"Loading scores: {args.baseline}, {args.finetuned}")
    baseline_scores = pd.read_parquet(args.baseline)
    finetuned_scores = pd.read_parquet(args.finetuned)
    print(f"Loading index: {args.index}")
    index_close = pd.read_parquet(args.index)

    results, report = run_analysis(baseline_scores, finetuned_scores, index_close,
                                   args.n_boot, args.seed)
    _print_summary(results)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(results.to_dict(orient="records"), indent=2, default=str))
    args.output_report.write_text(report)
    print(f"\nWrote {args.output_csv}\nWrote {args.output_json}\nWrote {args.output_report}")


if __name__ == "__main__":
    main()
