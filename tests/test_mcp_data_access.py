"""Unit tests for src/mcp_server/data_access.py.

House style: build inputs as inline pandas DataFrames (no file I/O), assert
structural invariants exactly. Reads of the real gitignored parquet files, if
any, are gated behind ``@pytest.mark.data`` (excluded in CI via ``-m "not data"``).
"""

import datetime as dt

import pandas as pd
import pytest

from src.mcp_server import data_access


def _scores_frame() -> pd.DataFrame:
    """Two tickers, each with ceo+cfo rows per call, mirroring the real schema.

    AAPL has two calls (2021-01-28, 2021-04-22); MSFT has one (2021-02-02).
    ``return_start_date`` is a plain ``datetime.date`` as in the scores parquet.
    """
    rows = [
        # AAPL 2021-01-28
        ("AAPL_2021-01-28", "AAPL", dt.date(2021, 1, 28), "ceo", 0.60, 0.30, 0.10),
        ("AAPL_2021-01-28", "AAPL", dt.date(2021, 1, 28), "cfo", 0.40, 0.50, 0.10),
        # AAPL 2021-04-22
        ("AAPL_2021-04-22", "AAPL", dt.date(2021, 4, 22), "ceo", 0.20, 0.10, 0.70),
        ("AAPL_2021-04-22", "AAPL", dt.date(2021, 4, 22), "cfo", 0.30, 0.10, 0.60),
        # MSFT 2021-02-02
        ("MSFT_2021-02-02", "MSFT", dt.date(2021, 2, 2), "ceo", 0.10, 0.80, 0.10),
        ("MSFT_2021-02-02", "MSFT", dt.date(2021, 2, 2), "cfo", 0.20, 0.70, 0.10),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "transcript_id",
            "ticker",
            "return_start_date",
            "speaker",
            "prob_neutral",
            "prob_positive",
            "prob_negative",
        ],
    )


@pytest.fixture
def scores(monkeypatch):
    """Patch the read seam so data_access reads the inline frame, not a file."""
    frame = _scores_frame()
    monkeypatch.setattr(data_access, "_read_scores", lambda _path: frame.copy())
    return frame


_RESULT_COLUMNS = {
    "ticker",
    "return_start_date",
    "transcript_id",
    "prob_neutral",
    "prob_positive",
    "prob_negative",
}


# --- ticker_is_covered ------------------------------------------------------

def test_covered_ticker_is_true(scores):
    assert data_access.ticker_is_covered("AAPL") is True


def test_uncovered_ticker_is_false(scores):
    assert data_access.ticker_is_covered("ZZZZ") is False


def test_coverage_is_case_insensitive(scores):
    assert data_access.ticker_is_covered("aapl") is True


# --- search_ticker_scores: found -------------------------------------------

def test_found_ticker_aggregates_ceo_cfo_per_call(scores):
    out = data_access.search_ticker_scores("AAPL")

    # Two calls -> two aggregated rows (one per transcript_id / call date).
    assert len(out) == 2
    assert list(out["return_start_date"]) == [dt.date(2021, 1, 28), dt.date(2021, 4, 22)]

    # First call probs are the mean of its ceo+cfo rows.
    first = out.iloc[0]
    assert first["prob_neutral"] == pytest.approx(0.50)
    assert first["prob_positive"] == pytest.approx(0.40)
    assert first["prob_negative"] == pytest.approx(0.10)

    assert set(out.columns) == _RESULT_COLUMNS


def test_search_is_case_insensitive(scores):
    out = data_access.search_ticker_scores("aapl")
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL"}


def test_full_history_default_returns_all_calls_sorted(scores):
    out = data_access.search_ticker_scores("AAPL")
    # No date range -> full history, ascending by call date.
    assert list(out["return_start_date"]) == sorted(out["return_start_date"])
    assert len(out) == 2


# --- search_ticker_scores: date filtering ----------------------------------

def test_date_range_is_inclusive_on_both_bounds(scores):
    out = data_access.search_ticker_scores(
        "AAPL", start_date="2021-01-28", end_date="2021-04-22"
    )
    assert len(out) == 2


def test_start_date_lower_bound_excludes_earlier_calls(scores):
    out = data_access.search_ticker_scores("AAPL", start_date="2021-02-01")
    assert list(out["return_start_date"]) == [dt.date(2021, 4, 22)]


@pytest.mark.parametrize("bad", ["01/28/2021", "2021-1-1", "2021-13-01", "garbage"])
def test_malformed_date_raises_clear_error(scores, bad):
    with pytest.raises(ValueError, match=r"start_date must be an ISO date"):
        data_access.search_ticker_scores("AAPL", start_date=bad)


def test_empty_range_match_returns_empty_but_ticker_covered(scores):
    out = data_access.search_ticker_scores(
        "AAPL", start_date="2022-01-01", end_date="2022-12-31"
    )
    assert out.empty
    # Covered ticker with no calls in range is distinct from never-covered.
    assert data_access.ticker_is_covered("AAPL") is True


def test_uncovered_ticker_search_returns_empty_frame(scores):
    out = data_access.search_ticker_scores("ZZZZ")
    assert out.empty
    # Empty frame still carries the stable column set.
    assert set(out.columns) == _RESULT_COLUMNS


# --- latest_scores_for_tickers ---------------------------------------------

def test_latest_returns_most_recent_row_per_ticker(scores):
    out = data_access.latest_scores_for_tickers(["AAPL", "MSFT"])

    # One row per covered ticker; AAPL collapses its two calls to the latest.
    assert list(out["ticker"]) == ["AAPL", "MSFT"]
    assert list(out["return_start_date"]) == [dt.date(2021, 4, 22), dt.date(2021, 2, 2)]
    assert set(out.columns) == _RESULT_COLUMNS


def test_latest_empty_input_returns_empty_stable_frame(scores):
    out = data_access.latest_scores_for_tickers([])
    assert out.empty
    assert set(out.columns) == _RESULT_COLUMNS


def test_latest_all_missing_returns_empty_stable_frame(scores):
    out = data_access.latest_scores_for_tickers(["ZZZZ", "QQQQ"])
    assert out.empty
    assert set(out.columns) == _RESULT_COLUMNS


def test_latest_dedupes_case_insensitive_first_seen_order(scores):
    out = data_access.latest_scores_for_tickers(["AAPL", "aapl", "MSFT"])
    assert list(out["ticker"]) == ["AAPL", "MSFT"]


def test_latest_partial_coverage_does_not_raise(scores):
    out = data_access.latest_scores_for_tickers(["AAPL", "ZZZZ"])
    # Only the covered ticker survives; the missing one is silently dropped.
    assert list(out["ticker"]) == ["AAPL"]


def test_latest_delegates_once_per_unique_ticker(monkeypatch, scores):
    calls = []

    real = data_access.search_ticker_scores

    def spy(ticker, *args, **kwargs):
        calls.append(ticker)
        return real(ticker, *args, **kwargs)

    monkeypatch.setattr(data_access, "search_ticker_scores", spy)
    data_access.latest_scores_for_tickers(["AAPL", "aapl", "MSFT"])
    # Deduped: one lookup per unique ticker, in first-seen order.
    assert calls == ["AAPL", "MSFT"]


# --- coverage_summary -------------------------------------------------------

def test_coverage_summary_full(scores):
    out = data_access.coverage_summary()

    # Two distinct tickers, sorted alphabetically, with per-ticker call counts.
    assert out["tickers"] == [
        {"ticker": "AAPL", "call_count": 2},
        {"ticker": "MSFT", "call_count": 1},
    ]
    assert out["covered_ticker_count"] == 2
    # Three distinct calls across the dataset (AAPL x2, MSFT x1).
    assert out["total_call_count"] == 3
    assert out["start_date"] == "2021-01-28"
    assert out["end_date"] == "2021-04-22"


def test_coverage_summary_prefix_filters_tickers(scores):
    out = data_access.coverage_summary(prefix="AA")
    assert [t["ticker"] for t in out["tickers"]] == ["AAPL"]
    assert out["covered_ticker_count"] == 1
    # Overall stats still describe the whole dataset, not the prefix subset.
    assert out["total_call_count"] == 3
    assert out["end_date"] == "2021-04-22"


def test_coverage_summary_prefix_is_case_insensitive(scores):
    out = data_access.coverage_summary(prefix="aa")
    assert [t["ticker"] for t in out["tickers"]] == ["AAPL"]


def test_coverage_summary_limit_caps_list_but_not_count(scores):
    out = data_access.coverage_summary(limit=1)
    assert [t["ticker"] for t in out["tickers"]] == ["AAPL"]
    # covered_ticker_count reflects the full match count, pre-limit.
    assert out["covered_ticker_count"] == 2


# --- integration: real parquet ---------------------------------------------

@pytest.mark.data
def test_search_real_finetuned_scores():
    path = data_access.FINETUNED_SCORES_PATH
    assert path.exists(), (
        f"Scores parquet not found at {path}. "
        "Run: python -m src.models.inference --mode finetuned"
    )

    assert data_access.ticker_is_covered("AAL") is True
    assert data_access.ticker_is_covered("ZZZZ") is False

    out = data_access.search_ticker_scores("AAL")
    assert not out.empty
    assert set(out.columns) == _RESULT_COLUMNS
    # One aggregated row per call: transcript_id unique in the result.
    assert out["transcript_id"].is_unique
