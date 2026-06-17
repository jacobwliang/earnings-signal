"""
ES-02 ingestion pipeline: load raw transcripts, clean, save to Parquet.

Pipeline steps (see build_clean):
  load_raw → parse_dates → drop_bad_dates → drop_duplicate_calls → clean_exchange

All DataFrame transformations live in src/data/utils.py.
"""
import sys
import pandas as pd
from pathlib import Path

# Allow `python3 src/data/ingest_transcripts.py` from any working directory.
_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.utils import (  # noqa: E402
    fix_list_dates,
    parse_dates,
    add_transcript_len,
    drop_bad_dates,
    drop_duplicate_calls,
    clean_exchange,
)

DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "motley-fool-data.pkl"
OUT_PATH  = _PROJECT_ROOT / "data" / "raw" / "transcripts.parquet"

REQUIRED_COLUMNS = [
    "date", "date_parsed", "call_hour", "after_close", "return_start_date",
    "is_international_hours", "ticker", "exchange", "exchange_clean",
    "q", "transcript", "transcript_len",
]


def load_raw() -> pd.DataFrame:
    """Load the raw Motley Fool pickle and apply the list-date structural fix."""
    df = pd.read_pickle(DATA_PATH)
    return fix_list_dates(df)


def build_clean() -> pd.DataFrame:
    """
    Run the full cleaning pipeline and return a DataFrame ready for Parquet.

      1. load_raw          — read pickle, fix list-type date cells
      2. add_transcript_len — character count needed before dedup
      3. parse_dates        — date_parsed, call_hour, after_close, return_start_date
      4. drop_bad_dates     — remove null/unparseable date rows (logged)
      5. drop_duplicate_calls — keep longest transcript per (ticker, date_parsed)
      6. clean_exchange     — strip leading '(' from exchange values
    """
    print("=== ES-02: transcript ingestion & cleaning ===")
    df = load_raw()
    print(f"Loaded {len(df):,} rows")

    df = add_transcript_len(df)
    df = parse_dates(df)
    print(f"[build_clean] International-hours calls (0-6 ET): {df['is_international_hours'].sum()}")

    df = drop_bad_dates(df)
    df = drop_duplicate_calls(df)
    df = clean_exchange(df)

    print(f"=== Final row count: {len(df):,} ===")
    return df


def save_parquet(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    """Validate required columns and write the cleaned DataFrame to Parquet."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns before save: {missing}")

    out = df[REQUIRED_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"[save_parquet] Wrote {len(out):,} rows → {path}")


def inspect(df: pd.DataFrame) -> None:
    """
    Exploratory audit of a raw or partially-cleaned transcript DataFrame.
    Intended for interactive use during development, not called by build_clean.
    """
    print("\n=== Shape ===")
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n=== Columns & dtypes ===")
    print(df.dtypes.to_string())

    print("\n=== Null counts ===")
    print(df.isnull().sum().to_string())

    print("\n=== List-type cells ===")
    print(f"List-type tickers: {df['ticker'].apply(lambda x: isinstance(x, list)).sum()}")
    print(f"List-type dates:   {df['date'].apply(lambda x: isinstance(x, list)).sum()}")

    print("\n=== Duplicates (ticker, date) ===")
    print(f"  Duplicate (ticker, date) pairs: {df.duplicated(subset=['ticker', 'date']).sum()}")

    print("\n=== Transcript length distribution ===")
    tl = df["transcript"].str.len()
    print(tl.describe().to_string())
    print(f"  Transcripts under 500 chars: {(tl < 500).sum()}")

    print("\n=== Exchange breakdown (raw) ===")
    print(df["exchange"].str.split(":").str[0].str.strip().value_counts().to_string())

    print("\n=== Calls per year ===")
    tmp = parse_dates(df)
    print(f"  Unparseable dates: {tmp['date_parsed'].isna().sum()}")
    print(tmp.groupby(tmp["date_parsed"].dt.year).size().to_string())

    print("\n=== After-hours split ===")
    print(tmp["after_close"].value_counts().to_string())
    print("\n  Call hour distribution:")
    print(tmp["call_hour"].value_counts().sort_index().to_string())
    print(f"\n  International-hours calls (0-6 ET): {(tmp['call_hour'] < 7).sum()}")


if __name__ == "__main__":
    save_parquet(build_clean())
