import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.compute_returns import compute_market_returns, get_price_on_date

RETURNS_PATH = Path("data/raw/returns.parquet")
TRANSCRIPTS_PATH = Path("data/raw/transcripts.parquet")

EXPECTED_COLUMNS = ["ticker", "return_start_date", "price_t0", "return_1d", "return_5d"]
# Sanity bounds for a 1-to-5 business-day forward return: a real move should not
# halve a stock to zero (-1.0) or 6x it (5.0) over a handful of trading days.
RETURN_LOWER_BOUND = -1.0
RETURN_UPPER_BOUND = 5.0


# --------------------------------------------------------------------------- #
# compute_market_returns — pure unit tests (no dataset needed, run in CI)
# --------------------------------------------------------------------------- #
def _spy_frame(dates, prices):
    """Build a single-column SPY close frame indexed by normalized dates."""
    idx = pd.to_datetime(dates).normalize()
    return pd.DataFrame({"SPY": prices}, index=idx)


def test_market_returns_anchor_on_same_bdays_as_compute_returns():
    # A run of trading days around Wed 2021-01-13. For return_start_date
    # 2021-01-13: price_t0 is the business day before (Tue 01-12), t1 the day
    # after (Thu 01-14), t5 five business days after (Wed 01-20 — skips the
    # 16/17 weekend). Prices chosen so the returns are exact round numbers.
    dates = ["2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15",
             "2021-01-19", "2021-01-20"]  # 18th is MLK but treated as a normal index day here
    prices = [100.0, 111.0, 110.0, 120.0, 130.0, 150.0]
    spy = _spy_frame(dates, prices)

    out = compute_market_returns(spy, ["2021-01-13"])

    assert list(out["return_start_date"]) == [pd.Timestamp("2021-01-13")]
    row = out.iloc[0]
    # price_t0 = 100 (01-12), t1 = 110 (01-14), t5 = 150 (01-20)
    assert np.isclose(row["market_return_1d"], (110.0 - 100.0) / 100.0)
    assert np.isclose(row["market_return_5d"], (150.0 - 100.0) / 100.0)
    # matches the same anchoring get_price_on_date would resolve directly
    assert get_price_on_date(spy, "SPY", pd.Timestamp("2021-01-12")) == 100.0


def test_market_returns_none_when_endpoint_price_missing():
    # Only price_t0 and t1 present; the t5 business day (01-20) is absent.
    dates = ["2021-01-12", "2021-01-13", "2021-01-14"]
    spy = _spy_frame(dates, [100.0, 111.0, 110.0])

    out = compute_market_returns(spy, ["2021-01-13"])
    row = out.iloc[0]
    assert np.isclose(row["market_return_1d"], 0.10)
    assert row["market_return_5d"] is None or pd.isna(row["market_return_5d"])


def test_market_returns_one_row_per_unique_date():
    dates = ["2021-01-12", "2021-01-13", "2021-01-14"]
    spy = _spy_frame(dates, [100.0, 111.0, 110.0])
    out = compute_market_returns(spy, ["2021-01-13", "2021-01-13"])
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# returns.parquet invariants — read the real (gitignored) dataset
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_returns_parquet_exists_with_expected_schema():
    assert RETURNS_PATH.exists(), f"{RETURNS_PATH} not found"
    df = pd.read_parquet(RETURNS_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )


@pytest.mark.data
def test_row_count_matches_transcripts():
    returns = pd.read_parquet(RETURNS_PATH)
    transcripts = pd.read_parquet(TRANSCRIPTS_PATH, columns=["ticker"])
    assert len(returns) == len(transcripts), (
        f"Returns has {len(returns)} rows but transcripts has {len(transcripts)} — no rows should be dropped"
    )


@pytest.mark.data
def test_non_null_return_implies_non_null_price_t0():
    df = pd.read_parquet(RETURNS_PATH)
    bad = df[df["return_1d"].notna() & df["price_t0"].isna()]
    assert bad.empty, (
        f"{len(bad)} rows have a return_1d but no price_t0 — indicates a computation error"
    )


@pytest.mark.data
def test_returns_are_finite_and_within_bounds():
    df = pd.read_parquet(RETURNS_PATH)
    for col in ("return_1d", "return_5d"):
        values = df[col].dropna()
        assert values.notna().all() and (values.abs() != float("inf")).all(), (
            f"{col} contains inf values"
        )
        out_of_range = values[(values < RETURN_LOWER_BOUND) | (values > RETURN_UPPER_BOUND)]
        assert out_of_range.empty, (
            f"{col} has {len(out_of_range)} values outside [{RETURN_LOWER_BOUND}, {RETURN_UPPER_BOUND}]"
        )


@pytest.mark.data
def test_aapl_has_a_non_null_return():
    df = pd.read_parquet(RETURNS_PATH)
    aapl = df[df["ticker"] == "AAPL"]
    assert not aapl.empty, "No AAPL rows found in returns"
    assert aapl["return_1d"].notna().any(), "Expected at least one AAPL row with a non-null return_1d"
