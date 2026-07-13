"""Sanity checks for MCP tool wiring in src/mcp_server/tools.py.

Verifies tools register against the FastMCP instance and that their pydantic
result schemas serialize, without exercising real data or models. The
data-access layer is mocked so no parquet is touched.
"""

import datetime as dt

import pandas as pd

from src.mcp_server import tools


def _latest_frame() -> pd.DataFrame:
    """One latest row per covered ticker, mirroring search_ticker_scores output."""
    rows = [
        ("AAPL_2021-04-22", "AAPL", dt.date(2021, 4, 22), 0.20, 0.10, 0.70),
        ("MSFT_2021-02-02", "MSFT", dt.date(2021, 2, 2), 0.10, 0.80, 0.10),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "transcript_id",
            "ticker",
            "return_start_date",
            "prob_neutral",
            "prob_positive",
            "prob_negative",
        ],
    )


def test_compare_tickers_covered_and_missing(monkeypatch):
    monkeypatch.setattr(
        tools, "latest_scores_for_tickers", lambda _tickers: _latest_frame()
    )

    result = tools.compare_tickers(["AAPL", "MSFT", "ZZZZ"])

    entries = {e.ticker: e for e in result.entries}
    assert list(entries) == ["AAPL", "MSFT", "ZZZZ"]

    # Covered ticker: populated latest, label is argmax of the latest row's probs.
    assert entries["AAPL"].ticker_covered is True
    assert entries["AAPL"].latest is not None
    assert entries["AAPL"].latest.label == "negative"
    assert entries["AAPL"].latest.earnings_date == "2021-04-22"

    assert entries["MSFT"].ticker_covered is True
    assert entries["MSFT"].latest.label == "positive"

    # Missing ticker: not covered, no latest.
    assert entries["ZZZZ"].ticker_covered is False
    assert entries["ZZZZ"].latest is None


def test_compare_tickers_dedupes_first_seen(monkeypatch):
    monkeypatch.setattr(
        tools, "latest_scores_for_tickers", lambda _tickers: _latest_frame()
    )

    result = tools.compare_tickers(["AAPL", "aapl", "MSFT"])
    assert [e.ticker for e in result.entries] == ["AAPL", "MSFT"]


def test_compare_tickers_result_serializes(monkeypatch):
    monkeypatch.setattr(
        tools, "latest_scores_for_tickers", lambda _tickers: _latest_frame()
    )

    dumped = tools.compare_tickers(["AAPL", "ZZZZ"]).model_dump()
    assert dumped["entries"][0]["ticker"] == "AAPL"
    assert dumped["entries"][1]["latest"] is None
