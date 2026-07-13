"""Market-cap subgroup analysis: does the sentiment/return correlation differ by size?

Splits calls into market-cap **terciles** (assigned cross-sectionally *within each
cohort*, e.g. the earnings-date quarter, so the size thresholds don't drift over
time) and reports a Spearman correlation with a bootstrap CI per tercile. Two
terciles are read as "differing" only when their CIs don't overlap — a plain
pairwise read, no p-value.

The statistic itself is not reimplemented here: :func:`spearman_ci` from
:mod:`src.analysis.correlate_returns` already gives a point Spearman rho with a
**ticker-clustered** bootstrap percentile CI. Reusing it keeps this analysis
consistent with the call-level primary analysis and means clustering (a ticker's
calls aren't independent) is handled, not merely caveated.

Prerequisite (out of scope here): a per-row market cap **as of the earnings date**.
The scored dataset does not carry one yet; sourcing it is a separate upstream step.
Every function below takes ``cap_col``/``cohort_col`` as an interface, so the
module is complete and tested against synthetic data independent of that step.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.analysis.correlate_returns import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    FINETUNED_SCORES_PATH,
    INDEX_PATH,
    ROOT,
    add_abnormal_returns,
    aggregate_to_call,
    spearman_ci,
)
from src.data.compute_returns import compute_market_returns

logger = logging.getLogger(__name__)

TERCILES = ("small", "mid", "large")

# Primary window first, then the three robustness windows (this is the row order
# written to the results files and quoted in the report).
SUBGROUP_RETURN_COLS = ("abn_return_1d", "return_1d", "abn_return_5d", "return_5d")

MARKET_CAPS_PATH = ROOT / "data" / "processed" / "market_caps.parquet"
OUTPUT_JSON_PATH = ROOT / "results" / "subgroup_market_cap_results.json"


def assign_cap_tercile(df: pd.DataFrame, cap_col: str, cohort_col: str) -> pd.Series:
    """Label each row ``'small'``/``'mid'``/``'large'`` by market-cap rank.

    Ranking is done **within each cohort** (e.g. same earnings-date quarter), not
    across the whole dataset, so a general drift in market caps over time can't
    push a later quarter's names into a different bucket. Returns a Series aligned
    to ``df.index``, NaN for rows with a missing cap or in a cohort with fewer than
    three names (too few to form terciles).
    """
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    for _, grp in df.groupby(cohort_col, sort=False):
        valid = grp[cap_col].dropna()
        if len(valid) < 3:
            continue
        # Rank first so ties get distinct positions and qcut always finds three
        # non-degenerate bin edges, even when many caps are equal.
        binned = pd.qcut(valid.rank(method="first"), 3, labels=list(TERCILES))
        labels.loc[valid.index] = binned.astype(object)
    return labels


def _ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float):
    """Whether two intervals overlap; NA if any bound is NaN (undefined CI)."""
    if any(pd.isna(v) for v in (a_low, a_high, b_low, b_high)):
        return pd.NA
    return a_low <= b_high and b_low <= a_high


def pairwise_tercile_differences(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise CI-overlap table from a :func:`subgroup_correlation_analysis` result.

    One row per tercile pair (small-mid, small-large, mid-large): ``ci_overlap``
    and its negation ``differ``. A missing tercile CI yields NA rather than a
    spurious "differs".
    """
    idx = summary.set_index("tercile")
    rows = []
    for a, b in (("small", "mid"), ("small", "large"), ("mid", "large")):
        ra, rb = idx.loc[a], idx.loc[b]
        overlap = _ci_overlap(ra["ci_low"], ra["ci_high"], rb["ci_low"], rb["ci_high"])
        rows.append({
            "pair": f"{a}-{b}",
            "ci_overlap": overlap,
            "differ": pd.NA if overlap is pd.NA else not overlap,
        })
    return pd.DataFrame(rows)


def subgroup_correlation_analysis(
    df: pd.DataFrame,
    sentiment_col: str,
    return_col: str,
    cap_col: str,
    cohort_col: str,
    ticker_col: str = "ticker",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    min_n: int = 30,
) -> pd.DataFrame:
    """Per-tercile Spearman rho + ticker-clustered bootstrap CI.

    Assigns terciles with :func:`assign_cap_tercile`, drops rows missing a
    tercile/sentiment/return, then runs :func:`spearman_ci` on each tercile.
    Returns one row per tercile: ``tercile | n | rho | ci_low | ci_high |
    significant | reliable`` (``reliable = n >= min_n``; a small-n tercile is
    still reported, just flagged). The row-drop count is logged and stored on
    ``result.attrs`` (``n_rows_total`` / ``n_rows_dropped``).
    """
    work = df.copy()
    work["_tercile"] = assign_cap_tercile(work, cap_col, cohort_col)

    n_total = len(work)
    work = work.dropna(subset=["_tercile", sentiment_col, return_col])
    n_dropped = n_total - len(work)
    logger.info(
        "subgroup: dropped %d/%d rows (missing cap/cohort/sentiment/return)",
        n_dropped, n_total,
    )

    rows = []
    for tercile in TERCILES:
        sub = work[work["_tercile"] == tercile]
        if len(sub) == 0:
            logger.warning("tercile %r has no rows after filtering", tercile)
            rows.append({
                "tercile": tercile, "n": 0, "rho": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "significant": False, "reliable": False,
            })
            continue
        res = spearman_ci(
            sub, sentiment_col, return_col, ticker_col=ticker_col,
            n_boot=n_boot, seed=seed,
        )
        if np.isnan(res["rho"]):
            logger.warning("tercile %r: correlation undefined (zero variance)", tercile)
        rows.append({
            "tercile": tercile, "n": res["n"], "rho": res["rho"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "significant": res["significant"], "reliable": res["n"] >= min_n,
        })

    summary = pd.DataFrame(rows)
    summary.attrs["n_rows_total"] = n_total
    summary.attrs["n_rows_dropped"] = n_dropped
    return summary


# --------------------------------------------------------------------------- #
# Script: build the call table, run every window, write the results JSON the
# report reads from. Run with: python -m src.analysis.subgroup_market_cap
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scores = pd.read_parquet(FINETUNED_SCORES_PATH)
    caps = pd.read_parquet(MARKET_CAPS_PATH)
    market_returns = compute_market_returns(
        pd.read_parquet(INDEX_PATH), scores["return_start_date"].unique())
    call = aggregate_to_call(add_abnormal_returns(scores, market_returns)).merge(
        caps[["transcript_id", "market_cap"]], on="transcript_id", how="left")
    call["cohort"] = pd.to_datetime(call["return_start_date"]).dt.to_period("Q").astype(str)

    tercile_rows, pair_rows = [], []
    for return_col in SUBGROUP_RETURN_COLS:
        summary = subgroup_correlation_analysis(
            call, "sentiment_score", return_col, "market_cap", "cohort")
        tercile_rows.append(summary.assign(return_col=return_col, kind="tercile"))
        pair_rows.append(pairwise_tercile_differences(summary)
                         .assign(return_col=return_col, kind="pair"))

    results = pd.concat([*tercile_rows, *pair_rows], ignore_index=True)
    results["n_boot"], results["seed"] = DEFAULT_N_BOOT, DEFAULT_SEED
    results = results.reindex(columns=[
        "return_col", "kind", "tercile", "n", "rho", "ci_low", "ci_high",
        "significant", "reliable", "pair", "ci_overlap", "differ", "n_boot", "seed"])

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(results.to_dict(orient="records"), indent=2, default=str))
    print(f"Wrote {OUTPUT_JSON_PATH}")
