import pandas as pd
import pytest

from src.data.join_master import (
    merge_master,
    validate_no_column_overlap,
    validate_unique_keys,
)


def test_no_column_overlap():
    """validate_no_column_overlap raises when a non-key column appears in both frames."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "return_start_date": ["2024-01-02"],
            "price_t0": [100.0],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "return_start_date": ["2024-01-02"],
            "price_t0": [100.0],
        }
    )
    with pytest.raises(ValueError):
        validate_no_column_overlap(transcripts, returns)


def test_duplicate_keys_raises():
    """validate_unique_keys raises when (ticker, return_start_date) is duplicated."""
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "return_start_date": ["2024-01-02", "2024-01-02"],
            "text": ["call one", "call two"],
        }
    )
    with pytest.raises(ValueError):
        validate_unique_keys(df, "transcripts")


def test_master_row_count():
    """merge_master keeps only matched rows: 3 transcripts, 2 matches -> 2 rows."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "return_start_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "date_parsed": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "text": ["aapl call", "msft call", "goog call"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "price_t0": [185.0, 370.0],
            "return_1d": [0.01, -0.02],
            "return_5d": [0.03, 0.01],
        }
    )
    master = merge_master(transcripts, returns)
    assert len(master) == 2
    assert 0 < len(master) < len(transcripts)


def test_master_unique_keys():
    """merge_master produces a result whose (ticker, return_start_date) keys are unique."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "date_parsed": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "text": ["aapl call", "msft call"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "price_t0": [185.0, 370.0],
            "return_1d": [0.01, -0.02],
            "return_5d": [0.03, 0.01],
        }
    )
    master = merge_master(transcripts, returns)
    assert not master.duplicated(subset=["ticker", "return_start_date"]).any()


def test_price_t0_not_null():
    """merge_master result has no null price_t0 when every matched return row has one."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "date_parsed": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "text": ["aapl call", "msft call"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "price_t0": [185.0, 370.0],
            "return_1d": [0.01, -0.02],
            "return_5d": [0.03, 0.01],
        }
    )
    master = merge_master(transcripts, returns)
    assert master["price_t0"].isna().sum() == 0


def test_return_null_rates():
    """merge_master carries through null returns: nulls present but not all rows null."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "return_start_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "date_parsed": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "text": ["aapl call", "msft call", "goog call"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "return_start_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "price_t0": [185.0, 370.0, 140.0],
            "return_1d": [0.01, None, 0.02],
            "return_5d": [None, 0.01, 0.03],
        }
    )
    master = merge_master(transcripts, returns)
    for col in ("return_1d", "return_5d"):
        null_count = master[col].isna().sum()
        assert 0 < null_count < len(master)


def test_date_parsed_present():
    """merge_master preserves date_parsed from transcripts with no nulls in the result."""
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "date_parsed": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "text": ["aapl call", "msft call"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "return_start_date": ["2024-01-02", "2024-01-03"],
            "price_t0": [185.0, 370.0],
            "return_1d": [0.01, -0.02],
            "return_5d": [0.03, 0.01],
        }
    )
    master = merge_master(transcripts, returns)
    assert "date_parsed" in master.columns
    assert master["date_parsed"].isna().sum() == 0
