"""Pydantic v2 result models for the MCP tools' structured returns.

FastMCP derives each tool's output JSON schema from these models. Concrete
models follow the chosen tool surface (e.g. an earnings-call sentiment result,
a transcript-search result).
"""

from typing import Literal

from pydantic import BaseModel


class EarningsCallResult(BaseModel):
    earnings_date: str
    label: Literal["neutral", "positive", "negative"]
    probabilities: dict[str, float]
    model_run_id: str
    coverage_flag: Literal["complete", "missing_price_data", "chunk_partial"]


class TickerSentimentHistoryResult(BaseModel):
    ticker: str
    start_date: str | None
    end_date: str | None
    ticker_covered: bool
    match_count: int
    results: list[EarningsCallResult]


class TickerComparisonEntry(BaseModel):
    ticker: str
    ticker_covered: bool
    latest: EarningsCallResult | None


class CompareTickersResult(BaseModel):
    entries: list[TickerComparisonEntry]


class SentimentClassification(BaseModel):
    label: Literal["neutral", "positive", "negative"]
    probabilities: dict[str, float]
    model_run_id: str


class TickerCoverage(BaseModel):
    ticker: str
    call_count: int


class CoverageSummaryResult(BaseModel):
    tickers: list[TickerCoverage]
    covered_ticker_count: int
    total_call_count: int
    start_date: str | None
    end_date: str | None


class TranscriptSection(BaseModel):
    speaker: Literal["ceo", "cfo", "other_exec"]
    text: str


class TranscriptResult(BaseModel):
    ticker: str
    earnings_date: str
    sections: list[TranscriptSection]
    truncated: bool
