"""Compute post-earnings forward returns for each transcript.

Step 2 of ES-03. Pure computation — no network calls. For every transcript row
this looks up the closing price the business day before return_start_date
(price_t0) and the closing prices 1 and 5 business days after, then computes the
forward returns relative to price_t0. Missing prices yield None rather than a
dropped row, so the output always has one record per transcript.

Prices are looked up on exact dates only: no forward fill, interpolation, or
imputation.
"""

import pandas as pd
from pathlib import Path

# Paths are resolved relative to the repository root so the script runs the same
# way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_PATH = ROOT / "data" / "raw" / "transcripts.parquet"
PRICES_PATH = ROOT / "data" / "raw" / "prices_raw.parquet"
RETURNS_PATH = ROOT / "data" / "raw" / "returns.parquet"


def load_data(transcripts_path: str, prices_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the transcripts and the flattened closing-price table.

    Slices prices["Close"] immediately so the returned price frame has a single
    column level (one column per ticker) instead of the (field, ticker)
    MultiIndex stored on disk.
    """
    transcripts = pd.read_parquet(transcripts_path)
    close_df = pd.read_parquet(prices_path)["Close"]
    return transcripts, close_df


def get_price_on_date(close_df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float | None:
    """Return the closing price for `ticker` on exactly `date`, else None.

    Returns None if the ticker has no column, the date is absent from the price
    index, or the value is NaN. The lookup is exact: tz is stripped and the date
    normalized to midnight to match the tz-naive price index, but no forward
    fill, interpolation, or fallback is applied.
    """
    if ticker not in close_df.columns:
        return None
    date = pd.Timestamp(date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    date = date.normalize()
    if date not in close_df.index:
        return None
    price = close_df.at[date, ticker]
    return None if pd.isna(price) else float(price)


def compute_returns(transcripts: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_t0 and forward 1d/5d returns for every transcript row.

    Dates are derived with business-day offsets around return_start_date:
    price_t0 one business day before, and the 1d/5d returns from one and five
    business days after. A return is only computed when both its price and
    price_t0 are present; otherwise it is None. One record is emitted per input
    row — no rows are dropped.
    """
    bday = pd.tseries.offsets.BDay()
    records = []
    for row in transcripts.itertuples(index=False):
        start = row.return_start_date
        price_t0 = get_price_on_date(close_df, row.ticker, start - bday * 1)
        price_t1 = get_price_on_date(close_df, row.ticker, start + bday * 1)
        price_t5 = get_price_on_date(close_df, row.ticker, start + bday * 5)

        return_1d = (price_t1 - price_t0) / price_t0 if price_t0 and price_t1 is not None else None
        return_5d = (price_t5 - price_t0) / price_t0 if price_t0 and price_t5 is not None else None

        records.append({
            "ticker": row.ticker,
            "return_start_date": start,
            "price_t0": price_t0,
            "return_1d": return_1d,
            "return_5d": return_5d,
        })
    return pd.DataFrame.from_records(records)


def save_output(df: pd.DataFrame, path: str) -> None:
    """Write the returns DataFrame to a Parquet file."""
    df.to_parquet(path)
    print(f"Saved {len(df)} returns to {path}")


def main() -> None:
    """Load inputs, compute returns, report null rates, and save the output."""
    transcripts, close_df = load_data(TRANSCRIPTS_PATH, PRICES_PATH)
    returns = compute_returns(transcripts, close_df)

    print(f"Computed returns for {len(returns)} transcripts")
    print(f"return_1d null rate: {returns['return_1d'].isna().mean():.1%}")
    print(f"return_5d null rate: {returns['return_5d'].isna().mean():.1%}")

    save_output(returns, RETURNS_PATH)


if __name__ == "__main__":
    main()
