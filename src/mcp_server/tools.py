"""``@mcp.tool()``-decorated functions exposed by the MCP server.

Thin wrappers only: each tool validates/serializes and delegates to
:mod:`data_access` (parquet reads/filters) or :mod:`scoring` (live FinBERT
scoring). No business logic inline.

STUB STEP (ES-17/18, scaffold): each tool returns a hardcoded mock so the
FastMCP wiring can be proven end to end. The real delegation to
:mod:`data_access`/:mod:`scoring` lands on the follow-up feature branches.
"""

from src.models.inference import LABELS

from .data_access import (
    FINETUNED_SCORES_PATH,
    search_ticker_scores,
    ticker_is_covered,
)
from .schemas import (
    CompareTickersResult,
    EarningsCallResult,
    SearchTranscriptsResult,
    SentimentClassification,
    TickerComparisonEntry,
)
from .server import mcp


@mcp.tool()
def search_transcripts(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> SearchTranscriptsResult:
    """Look up earnings-call sentiment classifications for a ticker.

    Searches the pipeline's persisted sentiment results for ``ticker``,
    optionally restricted to earnings dates within ``[start_date, end_date]``
    (inclusive, ISO ``YYYY-MM-DD``). Results are classifications correlated
    with the transcript's language — not return predictions or forecasts.

    The ``ticker_covered`` flag distinguishes two distinct "empty" cases that
    callers must not conflate:
      * ``ticker_covered=False`` — the ticker was never scraped/processed, so
        the pipeline has no data for it at all (absence of coverage).
      * ``ticker_covered=True`` with ``match_count=0`` — the ticker is covered,
        but no earnings calls fall inside the requested date range (absence of
        matches within a covered ticker).

    Args:
        ticker: Equity ticker symbol to look up (e.g. ``"AAPL"``).
        start_date: Optional inclusive lower bound on earnings date, ISO
            ``YYYY-MM-DD``. ``None`` leaves the range open on the low end.
        end_date: Optional inclusive upper bound on earnings date, ISO
            ``YYYY-MM-DD``. ``None`` leaves the range open on the high end.

    Returns:
        A :class:`SearchTranscriptsResult` echoing the query and carrying the
        matched per-call classifications.
    """
    df = search_ticker_scores(ticker, start_date, end_date)

    model_run_id = FINETUNED_SCORES_PATH.stem
    results = []
    for row in df.itertuples(index=False):
        probabilities = {label: float(getattr(row, f"prob_{label}")) for label in LABELS}
        results.append(
            EarningsCallResult(
                earnings_date=str(row.return_start_date),
                label=max(LABELS, key=probabilities.get),
                probabilities=probabilities,
                model_run_id=model_run_id,
                coverage_flag="complete",
            )
        )

    # Non-empty results imply coverage; only pay the extra read to distinguish
    # never-covered from covered-but-no-matches-in-range when results are empty.
    ticker_covered = bool(results) or ticker_is_covered(ticker)

    return SearchTranscriptsResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        ticker_covered=ticker_covered,
        match_count=len(results),
        results=results,
    )


@mcp.tool()
def compare_tickers(tickers: list[str]) -> CompareTickersResult:
    """Compare the latest earnings-call sentiment across several tickers.

    A batch convenience wrapper over :func:`search_transcripts`: for each
    ticker it surfaces the most recent classification (or ``None`` when the
    ticker is covered but has no calls, or is not covered at all). Each entry's
    ``ticker_covered`` flag carries the same never-scraped-vs-no-matches
    distinction documented on :func:`search_transcripts`.

    Args:
        tickers: Equity ticker symbols to compare (e.g. ``["AAPL", "MSFT"]``).

    Returns:
        A :class:`CompareTickersResult` with one entry per input ticker.
    """
    # STUB: hardcoded mock, no real logic yet (see feature/compare-tickers).
    return CompareTickersResult(
        entries=[
            TickerComparisonEntry(ticker=t, ticker_covered=True, latest=None)
            for t in tickers
        ]
    )


@mcp.tool()
def classify_earnings_sentiment(transcript_text: str) -> SentimentClassification:
    """Classify the financial sentiment of arbitrary transcript text.

    Runs the fine-tuned FinBERT checkpoint live over ``transcript_text``,
    reusing the project's cleaning/chunking and chunk-to-document aggregation
    so the result matches how the batch pipeline scores calls. The output is a
    sentiment classification correlated with the language used — it is not a
    return prediction or forecast.

    Args:
        transcript_text: Raw earnings-call (or other) text to classify.

    Returns:
        A :class:`SentimentClassification` with the argmax ``label`` and the
        full per-class ``probabilities``.
    """
    # STUB: hardcoded mock, no real logic yet (see feature/classify-sentiment).
    return SentimentClassification(
        label="neutral",
        probabilities={"neutral": 0.6, "positive": 0.25, "negative": 0.15},
        model_run_id="stub-run-id",
    )
