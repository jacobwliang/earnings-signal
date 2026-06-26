import pandas as pd
import pytest
from pathlib import Path

PRICES_PATH = Path("data/raw/prices_raw.parquet")
FAILURES_PATH = Path("data/raw/price_fetch_failures.parquet")
TRANSCRIPTS_PATH = Path("data/raw/transcripts.parquet")

pytestmark = pytest.mark.data


def test_prices_parquet_exists_and_readable():
    assert PRICES_PATH.exists(), f"{PRICES_PATH} not found"
    df = pd.read_parquet(PRICES_PATH)
    assert not df.empty


def test_close_column_multi_level_index():
    df = pd.read_parquet(PRICES_PATH)
    assert isinstance(df.columns, pd.MultiIndex), "Expected MultiIndex columns from yfinance"
    assert df.columns.nlevels == 2, f"Expected 2 column levels, got {df.columns.nlevels}"
    assert "Close" in df.columns.get_level_values(0), "'Close' not found in top-level column index"


def test_no_all_nan_close_prices():
    df = pd.read_parquet(PRICES_PATH)
    close = df["Close"]
    all_nan_cols = [col for col in close.columns if close[col].isna().all()]
    assert not all_nan_cols, (
        f"split_failures should have excluded these all-NaN tickers: {all_nan_cols}"
    )


def test_failures_parquet_exists_and_has_ticker_column():
    assert FAILURES_PATH.exists(), f"{FAILURES_PATH} not found"
    df = pd.read_parquet(FAILURES_PATH)
    assert list(df.columns) == ["ticker"], (
        f"Expected exactly ['ticker'] column, got {list(df.columns)}"
    )


def test_date_range_covers_transcripts():
    transcripts = pd.read_parquet(TRANSCRIPTS_PATH)
    return_dates = pd.to_datetime(transcripts["return_start_date"]).dt.tz_localize(None)

    prices = pd.read_parquet(PRICES_PATH)
    price_index = pd.to_datetime(prices.index)
    if price_index.tz is not None:
        price_index = price_index.tz_convert(None)

    expected_start = return_dates.min() - pd.Timedelta(days=10)
    expected_end = return_dates.max() + pd.Timedelta(days=10)

    assert price_index.min() <= expected_start, (
        f"Prices start {price_index.min().date()} is after required {expected_start.date()}"
    )
    # Allow up to 5 calendar days of slack: expected_end may land on a weekend/holiday,
    # so the last actual trading day can be up to 5 days earlier and still be valid.
    assert price_index.max() >= expected_end - pd.Timedelta(days=5), (
        f"Prices end {price_index.max().date()} is before required {expected_end.date()} (minus 5-day market-closure buffer)"
    )
