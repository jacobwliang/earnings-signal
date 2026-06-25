"""Recover price data for tickers that failed the original bulk download.

This is a standalone, idempotent recovery script. It does NOT re-run the bulk
download in download_prices.py. It reads the failures list and the existing
price data already on disk, retries each failed ticker one at a time (a slow
cadence that avoids re-triggering the rate limiting that caused the original
failures), and merges anything it recovers back into prices_raw.parquet.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Paths are resolved relative to the repository root so the script runs the same
# way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT / "data" / "raw" / "prices_raw.parquet"
FAILURES_PATH = ROOT / "data" / "raw" / "price_fetch_failures.parquet"

RETRY_DELAY_SECONDS = 10  # Per-ticker sleep between requests.


def load_failures(path: Path) -> list[str]:
    """Read the failures parquet and return the list of tickers to retry."""
    df = pd.read_parquet(path)
    return df["ticker"].tolist()


def derive_date_range(prices: pd.DataFrame) -> tuple[str, str]:
    """Derive the download window from the existing price data's index.

    Returns (start, end) as YYYY-MM-DD strings taken from the min and max of
    prices_raw's date index, so the retry uses the exact same range as the
    original download without hardcoding any dates.
    """
    start = pd.to_datetime(prices.index.min()).strftime("%Y-%m-%d")
    end = pd.to_datetime(prices.index.max()).strftime("%Y-%m-%d")
    return start, end


def normalize_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Force `data` into the (field, ticker) two-level column layout.

    A single-ticker yf.download may return either flat field columns or a
    (field, ticker) MultiIndex depending on the yfinance version. This rebuilds
    the columns so every recovered ticker matches prices_raw's structure exactly
    and concatenates cleanly along the column axis.
    """
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
    else:
        fields = data.columns
    data.columns = pd.MultiIndex.from_product([fields, [ticker]])
    return data


def retry_tickers(tickers: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Retry each failed ticker one at a time with a fixed delay between calls.

    Sleeps RETRY_DELAY_SECONDS before each request. A non-empty result is recorded as
    recovered; any exception or empty result lands the ticker in still_failed.
    A single bad ticker never aborts the loop.

    Returns (recovered, still_failed) where recovered maps ticker -> normalized
    price DataFrame.
    """
    recovered: dict[str, pd.DataFrame] = {}
    still_failed: list[str] = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"Retrying ticker {i + 1}/{total}...")
        time.sleep(RETRY_DELAY_SECONDS)
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as exc:  # noqa: BLE001 - never crash on one bad ticker
            print(f"  {ticker}: error ({exc!r}), still failed")
            still_failed.append(ticker)
            continue
        if data is not None and not data.empty:
            recovered[ticker] = normalize_columns(data, ticker)
        else:
            still_failed.append(ticker)
    return recovered, still_failed


def merge_recovered(prices: pd.DataFrame, recovered: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge recovered tickers into prices along the column axis.

    Drops any columns already present for a recovered ticker first so the result
    never contains duplicate ticker columns.
    """
    if not recovered:
        return prices
    recovered_tickers = set(recovered)
    keep_mask = ~prices.columns.get_level_values(1).isin(recovered_tickers)
    prices = prices.loc[:, keep_mask]
    merged = pd.concat([prices, *recovered.values()], axis=1)
    return merged.sort_index()


def main() -> None:
    """Run the recovery pass and overwrite the prices and failures parquets."""
    failures = load_failures(FAILURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    start, end = derive_date_range(prices)
    print(f"Retrying {len(failures)} failed tickers over {start} to {end}")

    recovered, still_failed = retry_tickers(failures, start, end)

    merged = merge_recovered(prices, recovered)
    merged.to_parquet(PRICES_PATH)
    pd.DataFrame({"ticker": still_failed}).to_parquet(FAILURES_PATH)

    print(f"Recovered {len(recovered)} tickers, {len(still_failed)} still failed")


if __name__ == "__main__":
    main()
