import pandas as pd
import yfinance as yf
from pathlib import Path

# Load tickers and return_start_date range from the cleaned transcripts Parquet.
def load_tickers_and_dates(path: str) -> tuple[list[str], str, str]:
    df = pd.read_parquet(path)
    tickers = df["ticker"].unique().tolist()
    start = (pd.to_datetime(df["return_start_date"]).min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(df["return_start_date"]).max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    return tickers, start, end

def download_bulk(tickers: list[str], start: str, end: str, batch_size: int = 100) -> pd.DataFrame:
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
    """
    Identifies tickers that came back empty from yf.download.
    prices["Close"] gives a DataFrame where each column is one ticker.
    A ticker failed if its column is missing entirely or is all NaN.
    """
    close_df = prices["Close"]  # shape: (dates, tickers)
    
    failed = []
    for ticker in tickers:
        if ticker not in close_df.columns:
            failed.append(ticker)
        elif close_df[ticker].isna().all():
            failed.append(ticker)
    
    valid = set(tickers) - set(failed)
    prices_clean = prices.loc[:, prices.columns.get_level_values(1).isin(valid)]
    print(f"{len(failed)} tickers failed, {len(valid)} valid")
    return prices_clean, failed

def save_outputs(prices: pd.DataFrame, failures: list[str], prices_path: str, failures_path: str) -> None:
    prices.to_parquet(prices_path)
    pd.DataFrame({"ticker": failures}).to_parquet(failures_path)
    print(f"Saved prices to {prices_path}")
    print(f"Saved {len(failures)} failures to {failures_path}")
    
def main():
    tickers, start, end = load_tickers_and_dates("../../data/raw/transcripts.parquet")
    print(f"Date range: {start} to {end}")
    prices = download_bulk(tickers, start, end)
    prices, failures = split_failures(prices, tickers)
    save_outputs(prices, failures, "../../data/raw/prices_raw.parquet", "../../data/raw/price_fetch_failures.parquet")

if __name__ == "__main__":
    main()