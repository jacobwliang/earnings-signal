"""Join transcripts with forward returns into the master dataset.

Loads the transcripts and the computed forward returns, validates that the two
frames share only the join keys and that those keys are unique in each, then
inner joins them on (ticker, return_start_date). The result keeps every
transcript column plus price_t0, return_1d, and return_5d from the returns side
and is written to data/processed/master.parquet.

Rows with a null price_t0 are dropped: without a starting price no forward
return can be computed for any horizon, so those rows carry no modeling value.
Null return_1d/return_5d values are still carried through unchanged on the rows
that survive — a row can have a valid price_t0 but a missing t+1 or t+5 price.
"""

import pandas as pd
from pathlib import Path

# Paths are resolved relative to the repository root so the script runs the same
# way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_PATH = ROOT / "data" / "raw" / "transcripts.parquet"
RETURNS_PATH = ROOT / "data" / "raw" / "returns.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
MASTER_PATH = PROCESSED_DIR / "master.parquet"

KEYS = ["ticker", "return_start_date"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the transcripts and returns parquets, returning (transcripts, returns)."""
    transcripts = pd.read_parquet(TRANSCRIPTS_PATH)
    returns = pd.read_parquet(RETURNS_PATH)
    return transcripts, returns


def validate_no_column_overlap(transcripts: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Raise if the two frames share any non-key column.

    Overlapping non-key columns would collide on merge (producing _x/_y suffixes)
    and signal that the returns side is carrying data it should not.
    """
    shared = (set(transcripts.columns) & set(returns.columns)) - set(KEYS)
    if shared:
        raise ValueError(
            "transcripts and returns share non-key columns: "
            f"{sorted(shared)}"
        )


def validate_unique_keys(df: pd.DataFrame, name: str) -> None:
    """Raise if (ticker, return_start_date) is not unique in df.

    A duplicate key in either input would fan out rows on the inner join, so the
    join is only well defined when both sides are unique on the keys.
    """
    duplicate_count = df.duplicated(subset=KEYS).sum()
    if duplicate_count:
        raise ValueError(
            f"{name} has {duplicate_count} duplicate (ticker, return_start_date) "
            "rows; keys must be unique before joining"
        )


def merge_master(transcripts: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Inner join transcripts with returns on (ticker, return_start_date).

    Only transcripts with a matching returns row survive; the result gains
    price_t0, return_1d, and return_5d from the returns side.
    """
    return transcripts.merge(returns, on=KEYS, how="inner")


def drop_null_price_t0(master: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a null price_t0.

    A missing starting price makes every forward return undefined, so these rows
    cannot contribute a label and are removed before the dataset is saved.
    """
    return master[master["price_t0"].notna()].reset_index(drop=True)


def log_stats(transcripts: pd.DataFrame, joined: pd.DataFrame, master: pd.DataFrame) -> None:
    """Print row counts, dropped rows, and null counts/rates for the returns.

    `joined` is the inner-join result before the null-price_t0 drop, so the two
    drop causes (no return match vs. null price_t0) can be reported separately.
    """
    n_transcripts = len(transcripts)
    n_joined = len(joined)
    n_master = len(master)
    print(f"Rows in transcripts: {n_transcripts}")
    print(f"Rows in master: {n_master}")
    print(f"Rows dropped (no return match): {n_transcripts - n_joined}")
    print(f"Rows dropped (null price_t0): {n_joined - n_master}")
    for col in ("return_1d", "return_5d"):
        null_count = master[col].isna().sum()
        null_rate = null_count / n_master if n_master else 0.0
        print(f"{col}: {null_count} nulls ({null_rate:.2%})")


def save_master(master: pd.DataFrame) -> None:
    """Write the master frame to data/processed/master.parquet, creating the dir."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_PATH, index=False)
    print(f"Wrote {len(master)} rows to {MASTER_PATH}")


def main() -> None:
    """Load, validate, join, log, and save the master dataset."""
    transcripts, returns = load_data()
    validate_no_column_overlap(transcripts, returns)
    validate_unique_keys(transcripts, "transcripts")
    validate_unique_keys(returns, "returns")
    joined = merge_master(transcripts, returns)
    master = drop_null_price_t0(joined)
    log_stats(transcripts, joined, master)
    save_master(master)


if __name__ == "__main__":
    main()
