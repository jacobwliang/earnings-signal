"""Unit tests for ES-06 transcript chunking.

All fixtures are built inline as DataFrames — no file I/O and no dependency on
the real (gitignored) dataset or a downloaded tokenizer. A whitespace
tokenizer stands in for BertTokenizer so token counts are exactly controllable:
one space-separated word == one token.
"""

import pandas as pd

from src.data.chunk_transcripts import (
    chunk_transcripts,
    slice_windows,
)


class WhitespaceTokenizer:
    """Deterministic stand-in for BertTokenizer: one token per word.

    ``encode`` splits on whitespace into word "ids"; ``decode`` joins them back.
    The chunking logic only slices and round-trips ids, so word strings serve as
    ids without affecting behavior.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def _make_df(**overrides):
    """One-row master_clean frame with sensible defaults, overridable per test."""
    row = {
        "ticker": "AAA",
        "return_start_date": pd.Timestamp("2020-01-15"),
        "section_parse_ok": True,
        "text_prepared_ceo": None,
        "text_prepared_cfo": None,
        "return_1d": 0.01,
        "return_5d": 0.05,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _text(n_tokens):
    """A string of exactly ``n_tokens`` whitespace-separated tokens."""
    return " ".join(["w"] * n_tokens)


def test_slice_windows_non_overlapping_and_drops_short():
    """510/510/min 20: a 1020-id list -> two full windows, a 1029-id list -> two
    full windows plus a dropped 9-id tail, a 1040-id list -> two + a kept 20."""
    assert [len(w) for w in slice_windows(list(range(1020)))] == [510, 510]
    assert [len(w) for w in slice_windows(list(range(1029)))] == [510, 510]
    assert [len(w) for w in slice_windows(list(range(1040)))] == [510, 510, 20]


def test_1020_tokens_two_chunks_per_speaker():
    """A transcript with 1,020 tokens produces exactly 2 chunks per speaker."""
    df = _make_df(text_prepared_ceo=_text(1020), text_prepared_cfo=_text(1020))
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert (out["speaker"] == "ceo").sum() == 2
    assert (out["speaker"] == "cfo").sum() == 2


def test_509_tokens_one_chunk():
    """A transcript with 509 tokens produces exactly 1 chunk."""
    df = _make_df(text_prepared_ceo=_text(509))
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert len(out) == 1
    assert out.iloc[0]["n_chunks"] == 1


def test_short_tail_chunk_dropped():
    """A tail chunk with fewer than 20 tokens is dropped (515 -> 1 chunk)."""
    df = _make_df(text_prepared_ceo=_text(515))
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert len(out) == 1  # the 5-token tail is dropped


def test_null_speaker_column_produces_no_rows():
    """A null speaker column produces no rows for that speaker."""
    df = _make_df(text_prepared_ceo=_text(1020), text_prepared_cfo=None)
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert (out["speaker"] == "cfo").sum() == 0
    assert (out["speaker"] == "ceo").sum() == 2


def test_section_parse_not_ok_produces_no_rows():
    """section_parse_ok == False produces no rows at all."""
    df = _make_df(
        section_parse_ok=False,
        text_prepared_ceo=_text(1020),
        text_prepared_cfo=_text(1020),
    )
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert len(out) == 0


def test_chunk_idx_zero_indexed_and_sequential():
    """chunk_idx is 0-indexed and sequential within each (transcript_id, speaker)."""
    df = _make_df(text_prepared_ceo=_text(2040), text_prepared_cfo=_text(1530))
    out = chunk_transcripts(df, WhitespaceTokenizer())
    for (_, _), grp in out.groupby(["transcript_id", "speaker"]):
        assert list(grp["chunk_idx"]) == list(range(len(grp)))


def test_n_chunks_matches_row_count_per_group():
    """n_chunks matches the actual number of rows for that (transcript_id, speaker)."""
    df = _make_df(text_prepared_ceo=_text(2040), text_prepared_cfo=_text(1020))
    out = chunk_transcripts(df, WhitespaceTokenizer())
    for (_, _), grp in out.groupby(["transcript_id", "speaker"]):
        assert (grp["n_chunks"] == len(grp)).all()


def test_returns_identical_across_chunks_of_a_transcript():
    """return_1d and return_5d are identical across all chunks for the same transcript."""
    df = _make_df(
        text_prepared_ceo=_text(2040),
        text_prepared_cfo=_text(2040),
        return_1d=0.0234,
        return_5d=-0.0456,
    )
    out = chunk_transcripts(df, WhitespaceTokenizer())
    assert out["return_1d"].nunique() == 1
    assert out["return_5d"].nunique() == 1
    assert out.iloc[0]["return_1d"] == 0.0234
    assert out.iloc[0]["return_5d"] == -0.0456


def test_transcript_id_format():
    """transcript_id equals ticker + "_" + return_start_date.astype(str) per row."""
    df = _make_df(
        ticker="MSFT",
        return_start_date=pd.Timestamp("2021-07-27 16:30", tz="America/New_York"),
        text_prepared_ceo=_text(1020),
    )
    out = chunk_transcripts(df, WhitespaceTokenizer())
    expected = out["ticker"] + "_" + out["return_start_date"].astype(str)
    assert (out["transcript_id"] == expected).all()
    assert out.iloc[0]["transcript_id"] == "MSFT_2021-07-27"
