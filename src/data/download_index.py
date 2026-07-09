"""Download the market benchmark (SPY) used to build abnormal returns.

The correlation analysis defines an abnormal return as ``stock_return -
market_return`` (beta = 1, same business-day window), which strips broad market
moves out of the raw forward returns so sentiment is tested against the
company-specific piece. This script fetches SPY adjusted close over the sample
window and writes it as a single-column ``SPY`` frame, ready to feed
``compute_market_returns`` in compute_returns.py.

One-time fetch, mirroring download_prices.py. Idempotent: it skips the download
when index_prices.parquet already covers the needed date range.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# Paths are resolved relative to the repository root so the script runs the same
# way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_PATH = ROOT / "data" / "raw" / "transcripts.parquet"
INDEX_PATH = ROOT / "data" / "raw" / "index_prices.parquet"

INDEX_TICKER = "SPY"
# Pad the window so price_t0 (one business day before the earliest call) and
# price_t5 (five business days after the latest call) are always in range.
START_PAD_DAYS = 10
END_PAD_DAYS = 15


def load_window(path: Path) -> tuple[str, str]:
    """Read the padded [start, end] download window from the transcripts Parquet.

    The window spans the return_start_date range, padded so the surrounding
    business-day anchoring points have price data to work with.
    """
    df = pd.read_parquet(path, columns=["return_start_date"])
    dates = pd.to_datetime(df["return_start_date"])
    start = (dates.min() - pd.Timedelta(days=START_PAD_DAYS)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=END_PAD_DAYS)).strftime("%Y-%m-%d")
    return start, end


def download_index(start: str, end: str, ticker: str = INDEX_TICKER) -> pd.DataFrame:
    """Download the benchmark's adjusted close as a single-column ``ticker`` frame.

    Returns a DataFrame indexed by a tz-naive, midnight-normalized DatetimeIndex
    with one column named ``ticker`` — the shape ``compute_market_returns`` and
    ``get_price_on_date`` expect (a close_df with one column per ticker). Raises if
    the download comes back empty.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"yf.download returned empty data for {ticker} ({start}..{end})")

    close = data["Close"]
    # Single-ticker downloads come back either as a Series or a one-column frame
    # depending on yfinance version; normalize both to a frame keyed by `ticker`.
    close = close.to_frame() if isinstance(close, pd.Series) else close
    close.columns = [ticker]

    index = pd.to_datetime(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    return close


def covers_window(path: Path, start: str, end: str) -> bool:
    """True if an existing index file already spans [start, end]."""
    if not path.exists():
        return False
    existing = pd.read_parquet(path)
    if existing.empty:
        return False
    idx = pd.to_datetime(existing.index)
    return idx.min() <= pd.Timestamp(start) and idx.max() >= pd.Timestamp(end)


def main() -> None:
    """Fetch the benchmark for the transcript date range and save it (idempotent)."""
    start, end = load_window(TRANSCRIPTS_PATH)
    print(f"Index window: {start} to {end}")

    if covers_window(INDEX_PATH, start, end):
        print(f"{INDEX_PATH} already covers the window — skipping download")
        return

    close = download_index(start, end)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(INDEX_PATH)
    print(f"Saved {len(close)} rows of {INDEX_TICKER} close to {INDEX_PATH}")


if __name__ == "__main__":
    main()
