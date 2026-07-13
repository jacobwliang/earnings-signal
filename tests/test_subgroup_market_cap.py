"""Unit tests for the market-cap subgroup analysis (inline DataFrames, no I/O)."""

import numpy as np
import pandas as pd

from src.analysis.subgroup_market_cap import (
    assign_cap_tercile,
    pairwise_tercile_differences,
    subgroup_correlation_analysis,
)


def _tercile_frame(slopes, n_per=60, noise=0.02, seed=0, na_cap=0):
    """Build a single-cohort frame whose caps split cleanly into three terciles.

    ``slopes`` maps 'small'/'mid'/'large' -> the sign/strength of the sentiment→
    return relationship for that tercile (Spearman is rank-based, so the *sign*
    and the noise level, not the magnitude, drive rho). Each row gets a unique
    ticker so the clustered bootstrap behaves like a row bootstrap (tight CIs).
    ``na_cap`` rows are appended with a missing cap so they fall out of every
    tercile.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for t_idx, (label, slope) in enumerate(slopes.items()):
        x = np.linspace(0.0, 1.0, n_per)
        y = slope * x + rng.normal(0.0, noise, n_per)
        # Offset caps by tercile so ranks land small < mid < large as labelled.
        caps = np.arange(n_per, dtype=float) + t_idx * 10_000
        frames.append(pd.DataFrame({
            "sentiment": x, "ret": y, "cap": caps, "cohort": "Q1",
            "ticker": [f"{label}{i}" for i in range(n_per)],
        }))
    if na_cap:
        frames.append(pd.DataFrame({
            "sentiment": rng.normal(0, 1, na_cap), "ret": rng.normal(0, 1, na_cap),
            "cap": [np.nan] * na_cap, "cohort": "Q1",
            "ticker": [f"NA{i}" for i in range(na_cap)],
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# assign_cap_tercile
# --------------------------------------------------------------------------- #
def test_assign_cap_tercile_within_cohort_and_min_size():
    df = pd.DataFrame({
        "cap": [1, 2, 3, 4, 5, 6, 100, np.nan],
        "cohort": ["A", "A", "A", "A", "A", "A", "B", "A"],
    })
    labels = assign_cap_tercile(df, "cap", "cohort")
    # Cohort A (6 valid names) → two per tercile, ordered by cap.
    assert list(labels.iloc[:6]) == ["small", "small", "mid", "mid", "large", "large"]
    # Cohort B has a single name (< 3) → NaN; the missing-cap row is NaN too.
    assert pd.isna(labels.iloc[6])
    assert pd.isna(labels.iloc[7])


# --------------------------------------------------------------------------- #
# subgroup_correlation_analysis
# --------------------------------------------------------------------------- #
def test_different_correlation_per_tercile_cis_do_not_overlap():
    # small strongly positive, large strongly negative → CIs must separate.
    df = _tercile_frame({"small": 1.0, "mid": 0.0, "large": -1.0}, noise=0.02, seed=1)
    summary = subgroup_correlation_analysis(df, "sentiment", "ret", "cap", "cohort",
                                            n_boot=300, seed=1)
    by = summary.set_index("tercile")
    assert by.loc["small", "rho"] > 0.8
    assert by.loc["large", "rho"] < -0.8

    pairs = pairwise_tercile_differences(summary).set_index("pair")
    assert bool(pairs.loc["small-large", "differ"]) is True
    assert bool(pairs.loc["small-large", "ci_overlap"]) is False


def test_same_correlation_per_tercile_cis_overlap():
    # Identical moderate relationship in all three → no false "differ".
    df = _tercile_frame({"small": 1.0, "mid": 1.0, "large": 1.0}, noise=0.5, seed=2)
    summary = subgroup_correlation_analysis(df, "sentiment", "ret", "cap", "cohort",
                                            n_boot=300, seed=2)
    pairs = pairwise_tercile_differences(summary)
    assert (pairs["ci_overlap"] == True).all()  # noqa: E712
    assert (pairs["differ"] == False).all()      # noqa: E712


def test_tercile_below_min_n_marked_unreliable():
    df = _tercile_frame({"small": 1.0, "mid": 1.0, "large": 1.0}, n_per=5, seed=3)
    summary = subgroup_correlation_analysis(df, "sentiment", "ret", "cap", "cohort",
                                            n_boot=50, seed=3, min_n=30)
    assert (summary["n"] == 5).all()
    assert (summary["reliable"] == False).all()  # noqa: E712  — no crash on small n


def test_zero_variance_tercile_returns_nan_without_crash():
    df = _tercile_frame({"small": 1.0, "mid": 1.0, "large": 1.0}, seed=4)
    # Flatten the small tercile's sentiment → correlation undefined there.
    df.loc[df["ticker"].str.startswith("small"), "sentiment"] = 0.5
    summary = subgroup_correlation_analysis(df, "sentiment", "ret", "cap", "cohort",
                                            n_boot=50, seed=4).set_index("tercile")
    assert np.isnan(summary.loc["small", "rho"])
    assert not np.isnan(summary.loc["mid", "rho"])  # other terciles unaffected


def test_missing_cap_rows_excluded_with_drop_count():
    df = _tercile_frame({"small": 1.0, "mid": 1.0, "large": 1.0}, n_per=40, na_cap=7,
                        seed=5)
    summary = subgroup_correlation_analysis(df, "sentiment", "ret", "cap", "cohort",
                                            n_boot=50, seed=5)
    # The 7 missing-cap rows get no tercile and are dropped; the 120 valid ones stay.
    assert summary.attrs["n_rows_total"] == 127
    assert summary.attrs["n_rows_dropped"] == 7
    assert summary["n"].sum() == 120
