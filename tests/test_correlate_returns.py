"""Unit tests for the correlation analysis (inline DataFrames, no file I/O)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlate_returns import (
    ROBUSTNESS_RETURN_COLS,
    add_abnormal_returns,
    aggregate_to_call,
    bootstrap_difference,
    robustness_table,
    spearman_ci,
)


def _speaker_row(tid, speaker, sentiment, n_chunks, ticker, date, r1, r5, a1, a5):
    return {
        "transcript_id": tid, "speaker": speaker, "sentiment_score": sentiment,
        "n_chunks": n_chunks, "ticker": ticker, "return_start_date": date,
        "return_1d": r1, "return_5d": r5, "abn_return_1d": a1, "abn_return_5d": a5,
    }


# --------------------------------------------------------------------------- #
# aggregate_to_call
# --------------------------------------------------------------------------- #
def test_aggregate_to_call_two_and_one_speaker():
    d_a = pd.Timestamp("2021-01-13")
    d_b = pd.Timestamp("2021-02-01")
    df = pd.DataFrame([
        _speaker_row("A", "ceo", 0.2, 3, "AA", d_a, 0.01, 0.02, 0.005, 0.01),
        _speaker_row("A", "cfo", 0.4, 1, "AA", d_a, 0.01, 0.02, 0.005, 0.01),
        _speaker_row("B", "ceo", -0.1, 2, "BB", d_b, -0.03, 0.0, -0.02, 0.0),
    ])

    out = aggregate_to_call(df).set_index("transcript_id")

    assert len(out) == 2
    # Two-speaker call: plain mean of sentiment, summed chunks, returns carried.
    assert np.isclose(out.loc["A", "sentiment_score"], 0.3)
    assert out.loc["A", "n_chunks"] == 4
    assert np.isclose(out.loc["A", "abn_return_1d"], 0.005)
    assert out.loc["A", "ticker"] == "AA"
    # Single-speaker call passes straight through — no special-casing.
    assert np.isclose(out.loc["B", "sentiment_score"], -0.1)
    assert out.loc["B", "n_chunks"] == 2


def test_aggregate_to_call_weight_by_chunks():
    d_a = pd.Timestamp("2021-01-13")
    df = pd.DataFrame([
        _speaker_row("A", "ceo", 0.2, 3, "AA", d_a, 0.01, 0.02, 0.005, 0.01),
        _speaker_row("A", "cfo", 0.4, 1, "AA", d_a, 0.01, 0.02, 0.005, 0.01),
    ])
    out = aggregate_to_call(df, weight_by_chunks=True).set_index("transcript_id")
    # (0.2*3 + 0.4*1) / (3 + 1) = 1.0 / 4 = 0.25
    assert np.isclose(out.loc["A", "sentiment_score"], 0.25)


# --------------------------------------------------------------------------- #
# add_abnormal_returns
# --------------------------------------------------------------------------- #
def test_add_abnormal_returns_subtracts_market():
    scores = pd.DataFrame([{
        "transcript_id": "A", "ticker": "AA",
        "return_start_date": pd.Timestamp("2021-01-13"),
        "return_1d": 0.03, "return_5d": 0.05,
    }])
    market = pd.DataFrame([{
        "return_start_date": pd.Timestamp("2021-01-13"),
        "market_return_1d": 0.01, "market_return_5d": 0.02,
    }])

    out = add_abnormal_returns(scores, market)
    assert np.isclose(out.loc[0, "abn_return_1d"], 0.02)
    assert np.isclose(out.loc[0, "abn_return_5d"], 0.03)


def test_add_abnormal_returns_aligns_date_object_keys():
    # scores store datetime.date; market stores Timestamp — the merge must align.
    import datetime
    scores = pd.DataFrame([{
        "transcript_id": "A", "ticker": "AA",
        "return_start_date": datetime.date(2021, 1, 13),
        "return_1d": 0.03, "return_5d": 0.05,
    }])
    market = pd.DataFrame([{
        "return_start_date": pd.Timestamp("2021-01-13"),
        "market_return_1d": 0.01, "market_return_5d": 0.02,
    }])
    out = add_abnormal_returns(scores, market)
    assert np.isclose(out.loc[0, "abn_return_1d"], 0.02)


# --------------------------------------------------------------------------- #
# spearman_ci
# --------------------------------------------------------------------------- #
def _monotonic_frame(n=200, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = x + rng.normal(0.0, noise, n)
    return pd.DataFrame({
        "sentiment_score": x, "r": y,
        "ticker": [f"T{i}" for i in range(n)],
    })


def test_spearman_ci_strong_positive_and_ci_brackets_rho():
    df = _monotonic_frame()
    res = spearman_ci(df, "sentiment_score", "r", n_boot=300, seed=1)
    assert res["n"] == len(df)
    assert res["rho"] > 0.9
    assert res["ci_low"] <= res["rho"] <= res["ci_high"]
    assert res["significant"] is True


def test_spearman_ci_deterministic_under_seed():
    df = _monotonic_frame()
    a = spearman_ci(df, "sentiment_score", "r", n_boot=300, seed=7)
    b = spearman_ci(df, "sentiment_score", "r", n_boot=300, seed=7)
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]


def test_spearman_ci_drops_null_pairs():
    df = _monotonic_frame(n=50)
    df.loc[0, "r"] = np.nan
    res = spearman_ci(df, "sentiment_score", "r", n_boot=50, seed=1)
    assert res["n"] == 49


# --------------------------------------------------------------------------- #
# bootstrap_difference
# --------------------------------------------------------------------------- #
def _paired_frames(n=150, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    ret = x + rng.normal(0.0, 0.02, n)
    ids = [f"C{i}" for i in range(n)]
    finetuned = pd.DataFrame({
        "transcript_id": ids, "ticker": ids, "sentiment_score": x, "abn_return_1d": ret,
    })
    baseline = pd.DataFrame({
        "transcript_id": ids, "ticker": ids,
        "sentiment_score": rng.normal(0.0, 1.0, n),  # uncorrelated with ret
        "abn_return_1d": ret,
    })
    return finetuned, baseline


def test_bootstrap_difference_finetuned_beats_baseline():
    ft, bl = _paired_frames()
    res = bootstrap_difference(ft, bl, "sentiment_score", "abn_return_1d", n_boot=400, seed=2)
    assert res["diff"] > 0
    assert res["ci_low"] > 0  # CI excludes zero on the positive side
    assert res["significant"] is True


def test_bootstrap_difference_requires_matching_transcripts():
    ft, bl = _paired_frames()
    with pytest.raises(AssertionError):
        bootstrap_difference(ft, bl.iloc[:-1], "sentiment_score", "abn_return_1d",
                             n_boot=10, seed=2)


# --------------------------------------------------------------------------- #
# robustness_table
# --------------------------------------------------------------------------- #
def test_robustness_table_shape_and_bh_monotonicity():
    rng = np.random.default_rng(0)
    n = 160
    x = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame({
        "sentiment_score": x,
        "return_1d": x + rng.normal(0, 0.1, n),
        "abn_return_1d": x + rng.normal(0, 0.1, n),
        "return_5d": rng.normal(0, 1, n),
        "abn_return_5d": rng.normal(0, 1, n),
        "ticker": [f"T{i % 40}" for i in range(n)],
    })
    table = robustness_table(df, n_boot=200, seed=3)
    assert list(table["return_col"]) == list(ROBUSTNESS_RETURN_COLS)
    assert len(table) == 4
    # BH-adjusted p-values are never below the raw p-values.
    assert (table["p_adjusted"] >= table["p_value"] - 1e-9).all()
