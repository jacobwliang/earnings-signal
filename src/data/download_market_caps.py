"""Source a per-call market cap **as of the earnings date** for the subgroup analysis.

The size-subgroup analysis (:mod:`src.analysis.subgroup_market_cap`) needs one
market cap per call, measured at the earnings date rather than today — otherwise a
company that has since grown or shrunk would be mis-bucketed. Market cap is not in
the scored dataset, so this one-time script builds it:

    market_cap = shares_outstanding(as of earnings date) x close(as of earnings date)

- **Shares** come from yfinance ``Ticker.get_shares_full`` — a point-in-time shares
  series — fetched per ticker (mirroring download_prices.py, failures collected).
- **Close** comes from the existing ``prices_raw.parquet`` (unadjusted close, so the
  product is an actual dollar market cap, not a split-adjusted one).
- Both are joined **as-of** (most recent value at or before the earnings date) via
  :func:`compute_market_caps`, which is pure and unit-tested independently of the
  network.

Output: ``data/processed/market_caps.parquet`` keyed by ``transcript_id`` (with
``ticker``/``return_start_date``), ready to merge onto the call-level frame as the
``cap_col`` the subgroup analysis expects.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = ROOT / "results" / "finetuned_scores.parquet"
PRICES_PATH = ROOT / "data" / "raw" / "prices_raw.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "market_caps.parquet"
FAILURES_PATH = ROOT / "data" / "processed" / "market_cap_fetch_failures.parquet"

# Pad the fetch window so shares reported just before the earliest call are still
# in range for the as-of (backward) lookup.
START_PAD_DAYS = 400
END_PAD_DAYS = 15
FETCH_SLEEP_SECONDS = 0.3  # Gentle pacing to stay under yfinance rate limits.


def load_call_keys(scores_path: Path) -> pd.DataFrame:
    """Unique (transcript_id, ticker, return_start_date) — one row per call."""
    df = pd.read_parquet(scores_path, columns=["transcript_id", "ticker", "return_start_date"])
    df = df.drop_duplicates(subset="transcript_id").reset_index(drop=True)
    df["return_start_date"] = pd.to_datetime(df["return_start_date"]).dt.normalize()
    return df


def close_long_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide ``prices_raw`` panel to long ``[ticker, date, close]``.

    ``prices_raw`` is a date-indexed frame with MultiIndex ``(field, ticker)``
    columns; we take the ``Close`` field and stack the tickers into rows.
    """
    close = prices["Close"]
    long = close.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    return long.dropna(subset=["close"])


def fetch_shares_outstanding(
    tickers, start: str, end: str, sleep: float = FETCH_SLEEP_SECONDS
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch point-in-time shares outstanding per ticker via ``get_shares_full``.

    Returns a long ``[ticker, date, shares_outstanding]`` frame and the list of
    tickers that came back empty or errored (kept, not raised, like the price
    downloader). Duplicate dates within a ticker keep the last reported value.
    """
    records, failures = [], []
    for i, ticker in enumerate(tickers, 1):
        try:
            series = yf.Ticker(ticker).get_shares_full(start=start, end=end)
        except Exception as exc:  # noqa: BLE001 — network/parse errors are per-ticker
            print(f"  [{i}/{len(tickers)}] {ticker}: fetch failed ({exc})")
            failures.append(ticker)
            continue
        if series is None or len(series) == 0:
            failures.append(ticker)
            continue

        idx = pd.to_datetime(series.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        rec = pd.DataFrame({
            "ticker": ticker,
            "date": idx.normalize(),
            "shares_outstanding": series.to_numpy(dtype=float),
        }).drop_duplicates(subset="date", keep="last")
        records.append(rec)
        time.sleep(sleep)

    if not records:
        shares = pd.DataFrame(columns=["ticker", "date", "shares_outstanding"])
    else:
        shares = pd.concat(records, ignore_index=True)
    return shares, failures


def _asof_merge(keys: pd.DataFrame, panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """As-of merge ``panel[value_col]`` onto ``keys`` by ticker, backward in time."""
    panel = panel.sort_values("date")
    merged = pd.merge_asof(
        keys, panel[["ticker", "date", value_col]],
        left_on="return_start_date", right_on="date",
        by="ticker", direction="backward",
    )
    return merged.drop(columns="date")


def compute_market_caps(
    keys: pd.DataFrame, shares: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    """Join shares and close as-of the earnings date and multiply → ``market_cap``.

    ``keys``: ``[transcript_id, ticker, return_start_date]``. ``shares``/``close``:
    long ``[ticker, date, <value>]``. For each call, the most recent shares and
    close at or before ``return_start_date`` are used (NaN market cap when either
    is unavailable before the date — the subgroup analysis drops those rows).
    """
    keys = keys.copy()
    keys["return_start_date"] = pd.to_datetime(keys["return_start_date"]).dt.normalize()
    keys = keys.sort_values("return_start_date").reset_index(drop=True)

    shares = shares.copy()
    shares["date"] = pd.to_datetime(shares["date"]).dt.normalize()
    close = close.copy()
    close["date"] = pd.to_datetime(close["date"]).dt.normalize()

    merged = _asof_merge(keys, shares, "shares_outstanding")
    merged = _asof_merge(merged, close, "close")
    merged["market_cap"] = merged["shares_outstanding"] * merged["close"]
    return merged


def main(argv=None) -> None:
    """Fetch shares, join earnings-date close, and write the per-call market caps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    parser.add_argument("--prices", type=Path, default=PRICES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--failures", type=Path, default=FAILURES_PATH)
    args = parser.parse_args(argv)

    keys = load_call_keys(args.scores)
    tickers = sorted(keys["ticker"].unique())
    start = (keys["return_start_date"].min() - pd.Timedelta(days=START_PAD_DAYS)).strftime("%Y-%m-%d")
    end = (keys["return_start_date"].max() + pd.Timedelta(days=END_PAD_DAYS)).strftime("%Y-%m-%d")
    print(f"{len(keys)} calls, {len(tickers)} tickers, shares window {start}..{end}")

    shares, failures = fetch_shares_outstanding(tickers, start, end)
    print(f"Fetched shares for {len(tickers) - len(failures)}/{len(tickers)} tickers "
          f"({len(failures)} failed)")

    close = close_long_from_prices(pd.read_parquet(args.prices))
    caps = compute_market_caps(keys, shares, close)

    covered = caps["market_cap"].notna().sum()
    print(f"Market cap resolved for {covered}/{len(caps)} calls")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    caps.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")
    if failures:
        pd.DataFrame({"ticker": failures}).to_parquet(args.failures, index=False)
        print(f"Wrote {len(failures)} failures to {args.failures}")


if __name__ == "__main__":
    main()
