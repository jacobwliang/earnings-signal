"""``@mcp.tool()``-decorated functions exposed by the MCP server.

Thin wrappers only: each tool validates/serializes and delegates to
:mod:`data_access` (parquet reads/filters) or :mod:`scoring` (live FinBERT
scoring). No business logic inline.
"""

from src.models.inference import LABELS

from .data_access import (
    FINETUNED_SCORES_PATH,
    coverage_summary,
    latest_scores_for_tickers,
    search_ticker_scores,
    ticker_is_covered,
)
from .schemas import (
    CompareTickersResult,
    CoverageSummaryResult,
    EarningsCallResult,
    SentimentClassification,
    TickerComparisonEntry,
    TickerCoverage,
    TickerSentimentHistoryResult,
)
from .scoring import MODEL_RUN_ID, classify_text
from .server import mcp


def _row_to_earnings_call_result(row, model_run_id: str) -> EarningsCallResult:
    """Build an :class:`EarningsCallResult` from one aggregated scores row.

    ``row`` is a namedtuple from ``DataFrame.itertuples(index=False)`` carrying
    ``return_start_date`` and the per-label ``prob_*`` columns.
    """
    probabilities = {label: float(getattr(row, f"prob_{label}")) for label in LABELS}
    return EarningsCallResult(
        earnings_date=str(row.return_start_date),
        label=max(LABELS, key=probabilities.get),
        probabilities=probabilities,
        model_run_id=model_run_id,
        coverage_flag="complete",
    )


@mcp.tool()
def get_ticker_sentiment_history(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> TickerSentimentHistoryResult:
    """Look up earnings-call sentiment classifications for a ticker.

    This is a keyed lookup by ticker (and optional date range) over the
    pipeline's persisted per-call sentiment — not a full-text search of
    transcript contents.

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
        A :class:`TickerSentimentHistoryResult` echoing the query and carrying the
        matched per-call classifications.
    """
    df = search_ticker_scores(ticker, start_date, end_date)

    model_run_id = FINETUNED_SCORES_PATH.stem
    results = [
        _row_to_earnings_call_result(row, model_run_id)
        for row in df.itertuples(index=False)
    ]

    # Non-empty results imply coverage; only pay the extra read to distinguish
    # never-covered from covered-but-no-matches-in-range when results are empty.
    ticker_covered = bool(results) or ticker_is_covered(ticker)

    return TickerSentimentHistoryResult(
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

    A batch convenience wrapper over :func:`get_ticker_sentiment_history`: for each
    ticker it surfaces the most recent classification (or ``None`` when the
    ticker is covered but has no calls, or is not covered at all). Each entry's
    ``ticker_covered`` flag carries the same never-scraped-vs-no-matches
    distinction documented on :func:`get_ticker_sentiment_history`.

    Args:
        tickers: Equity ticker symbols to compare (e.g. ``["AAPL", "MSFT"]``).

    Returns:
        A :class:`CompareTickersResult` with one entry per input ticker.
    """
    model_run_id = FINETUNED_SCORES_PATH.stem
    latest = latest_scores_for_tickers(tickers)
    # One latest row per covered ticker, keyed for case-insensitive lookup.
    by_ticker = {
        str(row.ticker).upper(): row for row in latest.itertuples(index=False)
    }

    # With no date range, coverage <=> presence in `latest` (covered <=> has
    # rows), so absence here means never-covered — no extra read needed.
    entries: list[TickerComparisonEntry] = []
    seen: set[str] = set()
    for ticker in tickers:
        key = ticker.strip().upper()
        if key in seen:
            continue
        seen.add(key)
        row = by_ticker.get(key)
        entries.append(
            TickerComparisonEntry(
                ticker=ticker,
                ticker_covered=row is not None,
                latest=(
                    _row_to_earnings_call_result(row, model_run_id)
                    if row is not None
                    else None
                ),
            )
        )

    return CompareTickersResult(entries=entries)


@mcp.tool()
def list_covered_tickers(
    prefix: str | None = None,
    limit: int | None = None,
) -> CoverageSummaryResult:
    """Summarize which tickers and earnings calls the pipeline covers.

    A discovery tool for "what do you have?" — reads the persisted sentiment
    scores once and reports the covered tickers (with per-ticker call counts),
    the overall earnings-date range, and total call count. Use this before the
    keyed lookups (:func:`get_ticker_sentiment_history`, :func:`compare_tickers`)
    when you don't yet know which tickers exist.

    Args:
        prefix: Optional case-insensitive ticker prefix filter (e.g. ``"AA"``
            returns only ``AAL``, ``AAPL``, ...). ``None`` returns all tickers.
        limit: Optional cap on the number of tickers listed, to avoid dumping
            hundreds of symbols. ``covered_ticker_count`` still reports the full
            count matching ``prefix`` so scope is visible even when truncated.

    Returns:
        A :class:`CoverageSummaryResult`. ``total_call_count`` and the date range
        describe the whole dataset and are unaffected by ``prefix``.
    """
    summary = coverage_summary(prefix=prefix, limit=limit)
    return CoverageSummaryResult(
        tickers=[TickerCoverage(**entry) for entry in summary["tickers"]],
        covered_ticker_count=summary["covered_ticker_count"],
        total_call_count=summary["total_call_count"],
        start_date=summary["start_date"],
        end_date=summary["end_date"],
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
    probabilities = classify_text(transcript_text)
    return SentimentClassification(
        label=max(LABELS, key=probabilities.get),
        probabilities=probabilities,
        model_run_id=MODEL_RUN_ID,
    )
