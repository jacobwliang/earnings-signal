"""ES-02 validation tests for the cleaned transcript Parquet file."""
import pandas as pd
import pytest
from pathlib import Path

PARQUET_PATH = Path(__file__).parents[1] / "data" / "raw" / "transcripts.parquet"

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def df():
    """Load the cleaned Parquet once for all tests in this module."""
    assert PARQUET_PATH.exists(), (
        f"Parquet file not found at {PARQUET_PATH}. "
        "Run: python3 src/data/ingest_transcripts.py"
    )
    return pd.read_parquet(PARQUET_PATH)


def test_parquet_exists():
    assert PARQUET_PATH.exists(), f"Missing: {PARQUET_PATH}"


def test_row_count_in_expected_range(df):
    assert 17_000 <= len(df) <= 18_755, (
        f"Row count {len(df):,} is outside expected range [17,000, 18,755]"
    )


def test_no_null_date_parsed(df):
    nulls = df["date_parsed"].isna().sum()
    assert nulls == 0, f"{nulls} null values in date_parsed"


def test_no_null_ticker(df):
    nulls = df["ticker"].isna().sum()
    assert nulls == 0, f"{nulls} null values in ticker"


def test_no_null_transcript(df):
    nulls = df["transcript"].isna().sum()
    assert nulls == 0, f"{nulls} null values in transcript"


def test_no_duplicate_ticker_date(df):
    dupes = df.duplicated(subset=["ticker", "date_parsed"]).sum()
    assert dupes == 0, f"{dupes} duplicate (ticker, date_parsed) pairs"


def test_return_start_date_no_lookahead(df):
    """return_start_date must never be before the call date (date only)."""
    call_date = df["date_parsed"].dt.normalize()
    # return_start_date is already date-only (normalized tz-aware), but stored
    # as tz-aware Timestamp — compare dates directly
    bad = (df["return_start_date"] < call_date).sum()
    assert bad == 0, f"{bad} rows have return_start_date before the call date"


def test_after_close_flag_consistency(df):
    """after_close must be True for every row where call_hour >= 16."""
    should_be_true = df["call_hour"] >= 16
    inconsistent = (should_be_true & ~df["after_close"]).sum()
    assert inconsistent == 0, (
        f"{inconsistent} rows have call_hour >= 16 but after_close is False"
    )


def test_required_columns_present(df):
    required = [
        "date", "date_parsed", "call_hour", "after_close", "return_start_date",
        "is_international_hours", "ticker", "exchange", "exchange_clean",
        "q", "transcript", "transcript_len",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
