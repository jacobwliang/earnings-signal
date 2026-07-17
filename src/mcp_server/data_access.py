"""Parquet load/filter helpers over the project's canonical data locations.

Plain functions returning ``pandas.DataFrame`` — no ``mcp`` dependency, so this
module stays pure and unit-testable with inline DataFrames.

NOTE: ``src/models/inference.py`` defines Colab-flat default paths (e.g.
``CHUNKS_PATH = HERE / "chunks.parquet"``) that do NOT point at the canonical
locations. This module deliberately defines its own correct path constants and
callers pass explicit paths rather than relying on those module defaults.

Canonical locations to expose (constants to be added at implementation):
    data/processed/chunks.parquet
    data/processed/master_clean.parquet
    results/inference/baseline_scores.parquet
    results/inference/finetuned_scores.parquet
    src/models/phrasebank_full_finetune/   (fine-tuned checkpoint dir)

Scaffolding step (ES-17/18): signatures only, each raising NotImplementedError.
:mod:`tools` does not call these yet — the stub step returns hardcoded mocks to
prove the FastMCP wiring; the real reads land on the feature branches noted per
function.
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from src.models.inference import PROB_COLUMNS

# Repo root is four levels up: src/mcp_server/data_access.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical, non-Colab paths (see module docstring on why these are redefined
# here rather than imported from src/models/inference.py).
CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.parquet"
MASTER_CLEAN_PATH = _REPO_ROOT / "data" / "processed" / "master_clean.parquet"
BASELINE_SCORES_PATH = _REPO_ROOT / "results" / "inference" / "baseline_scores.parquet"
FINETUNED_SCORES_PATH = _REPO_ROOT / "results" / "inference" / "finetuned_scores.parquet"

# Sentiment lives in the *_scores.parquet files keyed by (ticker,
# return_start_date). Real implementations import the label contract via
# `from src.models.inference import LABELS` rather than redefining it here.
#
# OPEN DECISION (deferred to feature/search-transcripts): which of baseline vs.
# fine-tuned scores this subsystem serves. The stub is agnostic; the follow-up
# branch picks one (likely finetuned) and threads it through explicitly.


# Result schema for search_ticker_scores: one aggregated row per earnings call.
_RESULT_COLUMNS = ["ticker", "return_start_date", "transcript_id"]


def _read_scores(scores_path: Path) -> pd.DataFrame:
    """Read a ``*_scores.parquet`` file. Single seam, monkeypatched in tests."""
    return pd.read_parquet(scores_path)


def _read_master_clean(master_path: Path) -> pd.DataFrame:
    """Read ``master_clean.parquet``. Single seam, monkeypatched in tests."""
    return pd.read_parquet(master_path)


# Public speaker keys -> the master_clean text columns backing get_transcript.
MASTER_SPEAKER_COLUMNS = {
    "ceo": "text_prepared_ceo",
    "cfo": "text_prepared_cfo",
    "other_exec": "text_prepared_other_exec",
}


def _match_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Case-insensitive filter of ``df`` to rows whose ``ticker`` matches."""
    return df[df["ticker"].str.upper() == ticker.strip().upper()]


def _parse_iso_date(value: str, field: str) -> dt.date:
    """Parse an ISO ``YYYY-MM-DD`` string, re-raising with a clear message.

    ``date.fromisoformat`` alone raises a low-level ``Invalid isoformat string``
    error; this wraps it so callers see which field was malformed and what
    format is expected.
    """
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from exc


def search_ticker_scores(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    scores_path: Path = FINETUNED_SCORES_PATH,
) -> pd.DataFrame:
    """Return one aggregated sentiment row per earnings call for ``ticker``.

    Filters ``scores_path`` (a ``*_scores.parquet`` file) to ``ticker``
    (case-insensitive) and, when given, to ``return_start_date`` in
    ``[start_date, end_date]`` (inclusive, ISO ``YYYY-MM-DD``). The two per-call
    rows (ceo + cfo) are collapsed to one row per call by an equal-weight mean
    over the ``prob_*`` columns. An empty frame means the ticker is covered but
    had no calls in range; callers distinguish never-scraped coverage via
    :func:`ticker_is_covered`.
    """
    df = _read_scores(scores_path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    matched = _match_ticker(df, ticker)

    call_dates = pd.to_datetime(matched["return_start_date"]).dt.date
    if start_date is not None:
        matched = matched[call_dates >= _parse_iso_date(start_date, "start_date")]
        call_dates = call_dates[matched.index]
    if end_date is not None:
        matched = matched[call_dates <= _parse_iso_date(end_date, "end_date")]

    if matched.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS + prob_cols)

    aggregated = (
        matched.groupby(["transcript_id", "return_start_date", "ticker"], sort=False)[
            prob_cols
        ]
        .mean()
        .reset_index()
        .sort_values("return_start_date")
        .reset_index(drop=True)
    )
    return aggregated[_RESULT_COLUMNS + prob_cols]


def ticker_is_covered(ticker: str, scores_path: Path = FINETUNED_SCORES_PATH) -> bool:
    """Whether ``ticker`` was ever scraped/processed into ``scores_path``.

    Distinguishes a never-covered ticker (no rows for it at all) from a covered
    ticker with no matches in a requested date range. Case-insensitive.
    """
    df = _read_scores(scores_path)
    return bool(_match_ticker(df, ticker).shape[0] > 0)


def coverage_summary(
    prefix: str | None = None,
    limit: int | None = None,
    scores_path: Path = FINETUNED_SCORES_PATH,
) -> dict:
    """Summarize which tickers/calls the scores parquet covers.

    Reads ``scores_path`` once and returns a dict with:
      * ``tickers`` — list of ``{"ticker", "call_count"}`` (distinct calls per
        ticker), alphabetically sorted, filtered by ``prefix`` (case-insensitive)
        and capped at ``limit``.
      * ``covered_ticker_count`` — total distinct tickers matching ``prefix``,
        *before* ``limit`` is applied, so callers can gauge scope.
      * ``total_call_count`` — distinct calls across the whole dataset.
      * ``start_date`` / ``end_date`` — overall ``return_start_date`` min/max
        (ISO ``YYYY-MM-DD``), or ``None`` when the dataset is empty.

    ``total_call_count`` and the date range describe overall coverage and are
    unaffected by ``prefix``; only the ticker list and ``covered_ticker_count``
    reflect the ``prefix`` filter.
    """
    df = _read_scores(scores_path)
    if df.empty:
        return {
            "tickers": [],
            "covered_ticker_count": 0,
            "total_call_count": 0,
            "start_date": None,
            "end_date": None,
        }

    tickers_upper = df["ticker"].str.upper()
    per_ticker = df.groupby(tickers_upper)["transcript_id"].nunique().sort_index()

    call_dates = pd.to_datetime(df["return_start_date"]).dt.date
    total_call_count = int(df["transcript_id"].nunique())
    start_date = call_dates.min().isoformat()
    end_date = call_dates.max().isoformat()

    if prefix:
        per_ticker = per_ticker[per_ticker.index.str.startswith(prefix.strip().upper())]
    covered_ticker_count = int(per_ticker.shape[0])

    if limit is not None:
        per_ticker = per_ticker.iloc[:limit]

    tickers = [
        {"ticker": str(t), "call_count": int(c)} for t, c in per_ticker.items()
    ]
    return {
        "tickers": tickers,
        "covered_ticker_count": covered_ticker_count,
        "total_call_count": total_call_count,
        "start_date": start_date,
        "end_date": end_date,
    }


def get_transcript_sections(
    ticker: str,
    earnings_date: str,
    speaker: str | None = None,
    master_path: Path = MASTER_CLEAN_PATH,
) -> dict[str, str]:
    """Return the prepared transcript section(s) for one earnings call.

    Looks up ``master_clean.parquet`` by ``(ticker, return_start_date)`` (the
    same join key used elsewhere; case-insensitive ticker, ISO ``earnings_date``)
    and returns a dict mapping speaker key -> section text for every non-empty
    section. When ``speaker`` is given, only that section is returned; otherwise
    all available sections (ceo, cfo, other_exec) are.

    Raises ``ValueError`` with a clear message when ``speaker`` is unknown, the
    date is malformed, or no call matches ``(ticker, earnings_date)``.
    """
    if speaker is not None and speaker not in MASTER_SPEAKER_COLUMNS:
        raise ValueError(
            f"speaker must be one of {sorted(MASTER_SPEAKER_COLUMNS)}, got {speaker!r}"
        )
    call_date = _parse_iso_date(earnings_date, "earnings_date")

    df = _read_master_clean(master_path)
    matched = _match_ticker(df, ticker)
    call_dates = pd.to_datetime(matched["return_start_date"]).dt.date
    matched = matched[call_dates == call_date]

    if matched.empty:
        raise ValueError(
            f"No transcript found for ticker {ticker.strip().upper()!r} "
            f"on {earnings_date}"
        )

    row = matched.iloc[0]
    columns = (
        {speaker: MASTER_SPEAKER_COLUMNS[speaker]}
        if speaker is not None
        else MASTER_SPEAKER_COLUMNS
    )
    sections: dict[str, str] = {}
    for spk, col in columns.items():
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            sections[spk] = str(value)
    return sections


def latest_scores_for_tickers(
    tickers: list[str],
    scores_path: Path = FINETUNED_SCORES_PATH,
) -> pd.DataFrame:
    """Return the most recent per-ticker sentiment row for each of ``tickers``.

    Batch backing for the ``compare_tickers`` tool: one row per covered ticker
    (its latest ``return_start_date``); never-covered tickers are omitted and
    surfaced as such by the tool layer. Case-insensitive duplicate tickers
    collapse to a single row, preserving first-seen order.

    Delegates to :func:`search_ticker_scores` per unique ticker (full history,
    no date range) and keeps the last row, which that function returns sorted
    ascending by call date. A never-covered ticker yields an empty frame there
    and is silently dropped, so partial coverage never fails the batch.
    """
    kept: list[pd.DataFrame] = []
    seen: set[str] = set()
    for ticker in tickers:
        key = ticker.strip().upper()
        if key in seen:
            continue
        seen.add(key)
        matched = search_ticker_scores(ticker, scores_path=scores_path)
        if not matched.empty:
            kept.append(matched.tail(1))

    if not kept:
        # Stable empty schema without touching disk (PROB_COLUMNS is the
        # canonical label contract; see module docstring).
        return pd.DataFrame(columns=_RESULT_COLUMNS + list(PROB_COLUMNS))

    return pd.concat(kept, ignore_index=True)
