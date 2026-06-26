import pandas as pd
import pytest
from pathlib import Path

RETURNS_PATH = Path("data/raw/returns.parquet")
TRANSCRIPTS_PATH = Path("data/raw/transcripts.parquet")

pytestmark = pytest.mark.data

EXPECTED_COLUMNS = ["ticker", "return_start_date", "price_t0", "return_1d", "return_5d"]
# Sanity bounds for a 1-to-5 business-day forward return: a real move should not
# halve a stock to zero (-1.0) or 6x it (5.0) over a handful of trading days.
RETURN_LOWER_BOUND = -1.0
RETURN_UPPER_BOUND = 5.0


def test_returns_parquet_exists_with_expected_schema():
    assert RETURNS_PATH.exists(), f"{RETURNS_PATH} not found"
    df = pd.read_parquet(RETURNS_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )


def test_row_count_matches_transcripts():
    returns = pd.read_parquet(RETURNS_PATH)
    transcripts = pd.read_parquet(TRANSCRIPTS_PATH, columns=["ticker"])
    assert len(returns) == len(transcripts), (
        f"Returns has {len(returns)} rows but transcripts has {len(transcripts)} — no rows should be dropped"
    )


def test_non_null_return_implies_non_null_price_t0():
    df = pd.read_parquet(RETURNS_PATH)
    bad = df[df["return_1d"].notna() & df["price_t0"].isna()]
    assert bad.empty, (
        f"{len(bad)} rows have a return_1d but no price_t0 — indicates a computation error"
    )


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


def test_aapl_has_a_non_null_return():
    df = pd.read_parquet(RETURNS_PATH)
    aapl = df[df["ticker"] == "AAPL"]
    assert not aapl.empty, "No AAPL rows found in returns"
    assert aapl["return_1d"].notna().any(), "Expected at least one AAPL row with a non-null return_1d"
