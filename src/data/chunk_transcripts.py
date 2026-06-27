"""06: Chunk per-speaker prepared remarks into FinBERT-sized token windows.

Reads master_clean.parquet (the per-speaker output of ES-05) and, for every
transcript that parsed cleanly, slices each of the CEO and CFO prepared-remarks
columns into non-overlapping 510-token windows — the usable content budget of
FinBERT's 512-token input once [CLS]/[SEP] are reserved. One row is emitted per
chunk so downstream inference can run a flat batch over chunks.parquet.

Each chunk carries its transcript's forward returns (return_1d / return_5d) so
chunk-level scores can be aggregated and joined back without re-reading master.

The tokenizer is injected (see ``chunk_transcripts``) so the chunking logic can
be unit-tested with a lightweight fake; ``main`` loads the real BertTokenizer.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MASTER_CLEAN_PATH = PROCESSED_DIR / "master_clean.parquet"
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"

WINDOW = 510
STRIDE = 510
MIN_TOKENS = 20
MODEL_NAME = "yiyanghkust/finbert-pretrain"

# (speaker label, source column) pairs, in output order.
_SPEAKERS = (
    ("ceo", "text_prepared_ceo"),
    ("cfo", "text_prepared_cfo"),
)

_OUTPUT_COLUMNS = (
    "transcript_id",
    "ticker",
    "return_start_date",
    "speaker",
    "chunk_idx",
    "chunk_text",
    "n_chunks",
    "return_1d",
    "return_5d",
)


def slice_windows(
    token_ids: list, window: int = WINDOW, stride: int = STRIDE,
    min_tokens: int = MIN_TOKENS,
) -> list[list]:
    """Slice ``token_ids`` into windows, dropping any window below ``min_tokens``.

    Windows start every ``stride`` ids and span ``window`` ids; with the default
    stride == window they are non-overlapping. A trailing window shorter than
    ``min_tokens`` (and, more generally, any too-short window) is dropped.
    """
    windows = []
    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start:start + window]
        if len(chunk) >= min_tokens:
            windows.append(chunk)
    return windows


def chunk_speaker_text(
    text: object, tokenizer, window: int = WINDOW, stride: int = STRIDE,
    min_tokens: int = MIN_TOKENS,
) -> list[str]:
    """Tokenize one speaker's text and return its decoded chunk strings.

    Null/non-string text yields no chunks. Text is tokenized with no special
    tokens and no truncation, sliced via :func:`slice_windows`, then each kept
    window is decoded back to a string with special tokens skipped.
    """
    if not isinstance(text, str):
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    windows = slice_windows(token_ids, window, stride, min_tokens)
    return [tokenizer.decode(w, skip_special_tokens=True) for w in windows]


def chunk_transcripts(
    df: pd.DataFrame, tokenizer, window: int = WINDOW, stride: int = STRIDE,
    min_tokens: int = MIN_TOKENS,
) -> pd.DataFrame:
    """Chunk every clean transcript's CEO/CFO remarks into one row per window.

    Rows with ``section_parse_ok == False`` are skipped entirely. ``return_start_date``
    is normalized to a plain date and ``transcript_id`` is ``ticker_<date>``;
    both feed the per-chunk rows so forward returns travel with each chunk.
    """
    return_dates = pd.to_datetime(df["return_start_date"]).dt.date

    rows = []
    for pos, (_, row) in enumerate(df.iterrows()):
        if not row["section_parse_ok"]:
            continue
        return_start_date = return_dates.iloc[pos]
        transcript_id = f"{row['ticker']}_{return_start_date}"
        for speaker, column in _SPEAKERS:
            chunks = chunk_speaker_text(
                row[column], tokenizer, window, stride, min_tokens
            )
            n_chunks = len(chunks)
            for chunk_idx, chunk_text in enumerate(chunks):
                rows.append({
                    "transcript_id": transcript_id,
                    "ticker": row["ticker"],
                    "return_start_date": return_start_date,
                    "speaker": speaker,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "n_chunks": n_chunks,
                    "return_1d": row["return_1d"],
                    "return_5d": row["return_5d"],
                })

    return pd.DataFrame(rows, columns=list(_OUTPUT_COLUMNS))


def log_stats(out: pd.DataFrame) -> None:
    """Print total rows, per-speaker counts/chunk stats, and empty-text checks."""
    print(f"Total rows: {len(out)}")
    print("Rows by speaker:")
    for speaker, count in out["speaker"].value_counts().items():
        print(f"  {speaker}: {count}")

    # n_chunks is per (transcript_id, speaker); collapse to one value per group.
    per_group = out.drop_duplicates(["transcript_id", "speaker"])
    print("n_chunks by speaker (mean / median):")
    for speaker, grp in per_group.groupby("speaker"):
        print(f"  {speaker}: mean={grp['n_chunks'].mean():.2f}  "
              f"median={grp['n_chunks'].median():.1f}")

    blank = out["chunk_text"].str.strip().eq("") | out["chunk_text"].isna()
    print(f"Empty/whitespace-only chunk_text rows: {int(blank.sum())}")


def main() -> None:
    """Chunk master_clean.parquet's CEO/CFO remarks and write chunks.parquet."""
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_parquet(MASTER_CLEAN_PATH)
    out = chunk_transcripts(df, tokenizer)

    log_stats(out)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CHUNKS_PATH, index=False)
    print(f"Wrote {len(out)} rows to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
