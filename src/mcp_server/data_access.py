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


def search_ticker_scores(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    scores_path: Path = FINETUNED_SCORES_PATH,
) -> pd.DataFrame:
    """Return the per-call sentiment rows for ``ticker`` within a date range.

    Filters ``scores_path`` (a ``*_scores.parquet`` file) to ``ticker`` and,
    when given, to ``return_start_date`` in ``[start_date, end_date]``
    (inclusive, ISO ``YYYY-MM-DD``). An empty frame means the ticker is covered
    but had no calls in range; callers distinguish never-scraped coverage via
    :func:`ticker_is_covered`.

    TODO(feature/search-transcripts): implement the parquet read + filter.
    """
    raise NotImplementedError("stub — see feature/search-transcripts")


def ticker_is_covered(ticker: str, scores_path: Path = FINETUNED_SCORES_PATH) -> bool:
    """Whether ``ticker`` was ever scraped/processed into ``scores_path``.

    Distinguishes a never-covered ticker (no rows for it at all) from a covered
    ticker with no matches in a requested date range.

    TODO(feature/search-transcripts): implement the coverage check.
    """
    raise NotImplementedError("stub — see feature/search-transcripts")


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
