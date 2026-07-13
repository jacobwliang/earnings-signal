"""Unit tests for the market-cap sourcing join (pure logic, no network)."""

import numpy as np
import pandas as pd

from src.data.download_market_caps import close_long_from_prices, compute_market_caps


def test_close_long_from_prices_reshapes_panel():
    idx = pd.DatetimeIndex(["2021-01-14", "2021-01-15"], name="Date")
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AA"), ("Close", "BB"), ("Volume", "AA"), ("Volume", "BB")],
        names=["Price", None],
    )
    prices = pd.DataFrame(
        [[10.0, 20.0, 100, 200], [11.0, np.nan, 110, 210]], index=idx, columns=cols
    )
    long = close_long_from_prices(prices).set_index(["ticker", "date"])
    assert long.loc[("AA", pd.Timestamp("2021-01-15")), "close"] == 11.0
    assert long.loc[("BB", pd.Timestamp("2021-01-14")), "close"] == 20.0
    # The NaN close (BB on 1/15) is dropped, not carried as a row.
    assert ("BB", pd.Timestamp("2021-01-15")) not in long.index


def test_compute_market_caps_uses_asof_values():
    keys = pd.DataFrame({
        "transcript_id": ["A", "B"],
        "ticker": ["AA", "AA"],
        "return_start_date": [pd.Timestamp("2021-01-15"), pd.Timestamp("2021-06-15")],
    })
    shares = pd.DataFrame({
        "ticker": ["AA", "AA"],
        "date": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-04-01")],
        "shares_outstanding": [1000.0, 1200.0],
    })
    close = pd.DataFrame({
        "ticker": ["AA", "AA", "AA"],
        "date": [pd.Timestamp("2021-01-14"), pd.Timestamp("2021-01-15"), pd.Timestamp("2021-06-14")],
        "close": [10.0, 11.0, 20.0],
    })
    out = compute_market_caps(keys, shares, close).set_index("transcript_id")
    # A: shares as-of 1/15 = 1000 (1/1 report), close as-of 1/15 = 11 → 11,000.
    assert out.loc["A", "market_cap"] == 1000.0 * 11.0
    # B: shares as-of 6/15 = 1200 (4/1 report), close as-of 6/15 = 20 (6/14) → 24,000.
    assert out.loc["B", "market_cap"] == 1200.0 * 20.0


def test_compute_market_caps_nan_when_no_prior_data():
    keys = pd.DataFrame({
        "transcript_id": ["A"],
        "ticker": ["AA"],
        "return_start_date": [pd.Timestamp("2021-01-15")],
    })
    # Shares only reported *after* the earnings date → no backward match.
    shares = pd.DataFrame({
        "ticker": ["AA"], "date": [pd.Timestamp("2021-02-01")],
        "shares_outstanding": [1000.0],
    })
    close = pd.DataFrame({
        "ticker": ["AA"], "date": [pd.Timestamp("2021-01-14")], "close": [10.0],
    })
    out = compute_market_caps(keys, shares, close)
    assert np.isnan(out.loc[0, "market_cap"])
