"""Bulk-download adjusted price data for every ticker in the transcripts set.

Reads the cleaned transcripts Parquet, derives the ticker list and date range,
and downloads prices in batches. Tickers that come back empty are then retried
one at a time on a slow cadence to recover rate-limit casualties, and the
results are written to prices_raw.parquet with the leftover failures alongside.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# Paths are resolved relative to the repository root so the script runs the same
# way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_PATH = ROOT / "data" / "raw" / "transcripts.parquet"
PRICES_PATH = ROOT / "data" / "raw" / "prices_raw.parquet"
FAILURES_PATH = ROOT / "data" / "raw" / "price_fetch_failures.parquet"

BATCH_SIZE = 100
RETRY_DELAY_SECONDS = 10.0  # Per-ticker sleep during the slow second pass.
COOLDOWN_SECONDS = 60       # Initial wait so the rate limiter resets before retrying.
BACKOFF_SECONDS = 30        # Extra wait when a retry is itself rate limited.


def load_tickers_and_dates(path: Path) -> tuple[list[str], str, str]:
    """Read the tickers and the download window from the transcripts Parquet.

    The window is padded by 10 days on each side of the return_start_date range
    so per-ticker returns have surrounding price data to work with.
    """
    df = pd.read_parquet(path)
    tickers = df["ticker"].unique().tolist()
    dates = pd.to_datetime(df["return_start_date"])
    start = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    return tickers, start, end


def download_bulk(tickers: list[str], start: str, end: str, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Download all tickers in batches and concatenate them column-wise.

    Returns a (field, ticker) MultiIndex-column DataFrame. Raises if every batch
    came back empty.
    """
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    frames = []
    for i, chunk in enumerate(chunks):
        print(f"Downloading batch {i + 1}/{len(chunks)} ({len(chunk)} tickers)...")
        batch = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False)
        if not batch.empty:
            frames.append(batch)
    if not frames:
        raise ValueError("yf.download returned empty DataFrame — check your tickers and date range")
    return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]


def split_failures(prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Split the bulk result into clean prices and the tickers that came back empty.

    prices["Close"] is a (dates, tickers) DataFrame; a ticker failed if its
    column is missing entirely or is all NaN.
    """
    close_df = prices["Close"]
    failed = [
        ticker for ticker in tickers
        if ticker not in close_df.columns or close_df[ticker].isna().all()
    ]
    valid = set(tickers) - set(failed)
    prices_clean = prices.loc[:, prices.columns.get_level_values(1).isin(valid)]
    print(f"{len(failed)} tickers failed, {len(valid)} valid")
    return prices_clean, failed


def retry_failures(failures: list[str], start: str, end: str, delay: float = RETRY_DELAY_SECONDS) -> tuple[pd.DataFrame, list[str]]:
    """Second-pass recovery for tickers that failed the bulk download.

    Many bulk failures are rate-limit casualties rather than genuinely
    unavailable symbols. This retries each failed ticker one at a time with a
    fixed delay so the slower cadence avoids re-triggering rate limiting.
    Genuinely unavailable tickers simply return empty again and are reported as
    still-failed.

    Returns (recovered_prices, still_failed): the recovered DataFrame shares the
    same (field, ticker) column structure as the bulk download.
    """
    recovered = {}
    still_failed = []
    # Let the rate limiter reset after the bulk session before hammering it again.
    time.sleep(COOLDOWN_SECONDS)
    for i, ticker in enumerate(failures):
        if i % 50 == 0:
            print(f"Retrying ticker {i + 1}/{len(failures)}...")
        time.sleep(delay)
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except YFRateLimitError:
            # Rate limited rather than unavailable — back off and keep it queued
            # as failed so it can be reconsidered, not dropped.
            print(f"Rate limited on {ticker}, backing off {BACKOFF_SECONDS}s...")
            time.sleep(BACKOFF_SECONDS)
            still_failed.append(ticker)
            continue
        if data is not None and not data.empty:
            recovered[ticker] = data
        else:
            still_failed.append(ticker)

    if not recovered:
        return pd.DataFrame(), still_failed
    return pd.concat(recovered.values(), axis=1), still_failed


def save_outputs(prices: pd.DataFrame, failures: list[str], prices_path: Path, failures_path: Path) -> None:
    """Write the price data and the failures list to their Parquet files."""
    prices.to_parquet(prices_path)
    pd.DataFrame({"ticker": failures}).to_parquet(failures_path)
    print(f"Saved prices to {prices_path}")
    print(f"Saved {len(failures)} failures to {failures_path}")


def main() -> None:
    """Run the bulk download, retry the failures, and save the merged result."""
    tickers, start, end = load_tickers_and_dates(TRANSCRIPTS_PATH)
    print(f"Date range: {start} to {end}")
    prices = download_bulk(tickers, start, end)
    prices, failures = split_failures(prices, tickers)
    save_outputs(prices, failures, PRICES_PATH, FAILURES_PATH)

    # Second pass: recover tickers that failed the bulk download due to rate
    # limiting rather than genuine unavailability.
    print(f"Retrying {len(failures)} failed tickers...")
    recovered, still_failed = retry_failures(failures, start, end)
    if not recovered.empty:
        prices = pd.concat([prices, recovered], axis=1)
    save_outputs(prices, still_failed, PRICES_PATH, FAILURES_PATH)
    print(f"Recovered {len(failures) - len(still_failed)} tickers, {len(still_failed)} still failed")


if __name__ == "__main__":
    main()
