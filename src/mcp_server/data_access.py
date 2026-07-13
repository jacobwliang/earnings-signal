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


def latest_scores_for_tickers(
    tickers: list[str],
    scores_path: Path = FINETUNED_SCORES_PATH,
) -> pd.DataFrame:
    """Return the most recent per-ticker sentiment row for each of ``tickers``.

    Batch backing for the ``compare_tickers`` tool: one row per covered ticker
    (its latest ``return_start_date``); never-covered tickers are omitted and
    surfaced as such by the tool layer.

    TODO(feature/compare-tickers): implement the grouped latest-row read.
    """
    raise NotImplementedError("stub — see feature/compare-tickers")
