"""
DataFrame transformation helpers for earnings transcript data.
All functions here are pure transformations (DataFrame in, DataFrame out)
and are reused across ingestion scripts (ES-02, ES-03, ...).
"""
import pandas as pd

MARKET_CLOSE_HOUR = 16  # 4 PM ET


def fix_list_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    379 raw rows have ``date`` stored as a list, e.g.
    ['Brunswick (BC 0.66%) Q4 2018 ', 'Jan. 31, 2019, 10:00 a.m. ET'].
    The real date string is always the second element.
    """
    mask = df["date"].apply(lambda x: isinstance(x, list))
    print(f"[fix_list_dates] Fixing {mask.sum()} list-type date rows")

    df = df.copy()
    df.loc[mask, "date"] = df.loc[mask, "date"].apply(
        lambda x: x[1].strip() if isinstance(x, list) and len(x) > 1 else None
    )

    still_bad = df["date"].apply(lambda x: isinstance(x, list)).sum()
    if still_bad:
        raise RuntimeError(f"{still_bad} list-type dates remain after fix — check raw data")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the raw ``date`` string column into structured datetime columns.

    Strips the trailing 'ET' suffix (unparseable by pandas), localizes to
    America/New_York so EST/EDT offsets are handled automatically, then adds:
      - ``date_parsed``            — tz-aware datetime in ET
      - ``call_hour``              — integer hour (0-23) in ET
      - ``after_close``            — True when call_hour >= 16
      - ``return_start_date``      — return window open date; next calendar day
                                     for after-close calls, same day otherwise
      - ``is_international_hours`` — True when call_hour < 7 (likely foreign HQ)
    """
    cleaned = df["date"].str.replace(r"\s*ET\s*$", "", regex=True)
    parsed = pd.to_datetime(cleaned, format="mixed", errors="coerce")
    parsed = parsed.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )

    df = df.copy()
    df["date_parsed"] = parsed
    df["call_hour"] = parsed.dt.hour
    df["after_close"] = parsed.dt.hour >= MARKET_CLOSE_HOUR
    df["is_international_hours"] = parsed.dt.hour < 7

    df["return_start_date"] = parsed.dt.normalize()
    df.loc[df["after_close"], "return_start_date"] = (
        df.loc[df["after_close"], "date_parsed"] + pd.Timedelta(days=1)
    ).dt.normalize()

    return df


def add_transcript_len(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``transcript_len`` — character count of each transcript."""
    df = df.copy()
    df["transcript_len"] = df["transcript"].str.len()
    return df


def drop_bad_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where ``date_parsed`` is NaT (null or unparseable).
    Logs each dropped ticker so no row is silently discarded.
    """
    bad = df[df["date_parsed"].isna()]
    if bad.empty:
        print("[drop_bad_dates] No null/unparseable dates found")
        return df

    print(f"[drop_bad_dates] Dropping {len(bad)} rows with null/unparseable dates:")
    for _, row in bad.iterrows():
        print(f"  ticker={row['ticker']}  raw_date={row['date']!r}")
    return df[df["date_parsed"].notna()].copy()


def drop_duplicate_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate (ticker, date_parsed) pairs, keeping the longest transcript.
    Logs a sample of duplicate groups and the total rows dropped.
    """
    dupes = df[df.duplicated(subset=["ticker", "date_parsed"], keep=False)]
    n_groups = dupes.groupby(["ticker", "date_parsed"]).ngroups
    print(f"[drop_duplicate_calls] Found {len(dupes)} rows in {n_groups} duplicate groups")

    if dupes.empty:
        return df

    sample_groups = list(dupes.groupby(["ticker", "date_parsed"]))[:3]
    print("[drop_duplicate_calls] Sample groups (ticker, date_parsed, transcript lengths):")
    for (ticker, dt), grp in sample_groups:
        print(f"  {ticker} {dt}  lengths={grp['transcript_len'].tolist()}")

    df_deduped = (
        df.sort_values("transcript_len", ascending=False)
        .drop_duplicates(subset=["ticker", "date_parsed"], keep="first")
    )
    print(f"[drop_duplicate_calls] Dropped {len(df) - len(df_deduped)} shorter duplicate rows")
    return df_deduped.copy()


def clean_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ``exchange_clean`` from the raw ``exchange`` field.

    Splits on ':' to remove the ticker suffix, then strips a leading '('
    present in 31 rows (e.g. '(NASDAQ:AAPL' → 'NASDAQ').
    """
    df = df.copy()
    raw_prefix = df["exchange"].str.split(":").str[0].str.strip()
    df["exchange_clean"] = raw_prefix.str.lstrip("(")
    dirty = (df["exchange_clean"] != raw_prefix).sum()
    print(f"[clean_exchange] Stripped leading '(' from {dirty} rows")
    return df
