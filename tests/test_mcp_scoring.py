"""Unit tests for src/mcp_server/scoring.py.

Fast tests use a hand-written fake tokenizer + a monkeypatched ``score_batch``
(see ``WhitespaceTokenizer``) rather than unittest.mock, so no checkpoint
download or forward pass is needed — they exercise the chunking and
chunk-to-document aggregation logic only. The one test that loads the real
FinBERT checkpoint is gated behind the ``model`` marker (excluded in CI).
"""

import numpy as np
import pytest

from src.mcp_server import scoring
from src.models.inference import FINETUNED_MODEL_PATH, LABELS


class WhitespaceTokenizer:
    """Deterministic stand-in for BertTokenizer: one token per word.

    ``encode`` splits on whitespace into word "ids"; ``decode`` joins them back.
    ``chunk_speaker_text`` only slices and round-trips ids, so word strings serve
    as ids without affecting behavior — one space-separated word == one token.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def _text(n_tokens):
    """A string of exactly ``n_tokens`` whitespace-separated tokens."""
    return " ".join(["w"] * n_tokens)


@pytest.fixture
def fake_model(monkeypatch):
    """Patch the checkpoint loader so no real model/tokenizer is needed.

    Returns the WhitespaceTokenizer so chunk boundaries are exactly controllable;
    the model/device are unused because ``score_batch`` is patched per-test.
    """
    tokenizer = WhitespaceTokenizer()
    monkeypatch.setattr(scoring, "_load_finetuned", lambda: (tokenizer, None, "cpu"))
    return tokenizer


# --- unit: chunking drives scoring ------------------------------------------

@pytest.mark.parametrize(
    "n_tokens, expected_chunks",
    [
        (1020, 2),  # two full 510-token windows
        (509, 1),   # one window
        (515, 1),   # one window; the 5-token tail is below MIN_TOKENS and dropped
    ],
)
def test_chunking_drives_scoring(fake_model, monkeypatch, n_tokens, expected_chunks):
    seen = {}

    def spy_score_batch(texts, tokenizer, model, device):
        seen["texts"] = texts
        return np.tile([0.2, 0.5, 0.3], (len(texts), 1))

    monkeypatch.setattr(scoring, "score_batch", spy_score_batch)

    scoring.classify_text(_text(n_tokens))
    assert len(seen["texts"]) == expected_chunks


# --- unit: aggregation is the equal-weight mean per class -------------------

def test_aggregation_equal_weight_mean(fake_model, monkeypatch):
    # Two chunks (1020 tokens) with distinct per-class probs; the document result
    # is their per-class mean: (0.2+0.4)/2, (0.7+0.5)/2, (0.1+0.1)/2.
    monkeypatch.setattr(
        scoring,
        "score_batch",
        lambda texts, *a: np.array([[0.2, 0.7, 0.1], [0.4, 0.5, 0.1]]),
    )

    result = scoring.classify_text(_text(1020))

    assert list(result) == list(LABELS)
    assert result["neutral"] == pytest.approx(0.3)
    assert result["positive"] == pytest.approx(0.6)
    assert result["negative"] == pytest.approx(0.1)
    assert sum(result.values()) == pytest.approx(1.0)


def test_single_chunk_passes_through(fake_model, monkeypatch):
    monkeypatch.setattr(
        scoring, "score_batch", lambda texts, *a: np.array([[0.15, 0.7, 0.15]])
    )

    result = scoring.classify_text(_text(509))
    assert result == pytest.approx({"neutral": 0.15, "positive": 0.7, "negative": 0.15})


# --- unit: empty / too-short input raises before any scoring ----------------

@pytest.mark.parametrize("bad", ["", "   ", _text(19)])
def test_no_scoreable_chunks_raises(fake_model, monkeypatch, bad):
    # score_batch must never be reached for un-chunkable input.
    def fail(*a, **k):
        raise AssertionError("score_batch should not be called for empty input")

    monkeypatch.setattr(scoring, "score_batch", fail)

    with pytest.raises(ValueError, match="no scoreable chunks"):
        scoring.classify_text(bad)


# --- integration: real checkpoint -------------------------------------------

@pytest.mark.model
def test_classify_real_checkpoint_shape_and_sum():
    assert FINETUNED_MODEL_PATH.exists(), (
        f"Checkpoint not found at {FINETUNED_MODEL_PATH}. "
        "Run: python -m src.models.finetune_phrasebank"
    )
    scoring._load_finetuned.cache_clear()

    result = scoring.classify_text(
        "Revenue grew 20% year over year and operating margins expanded. "
        "Management is very optimistic about demand heading into next quarter."
    )

    assert set(result) == set(LABELS)
    assert all(0.0 <= p <= 1.0 for p in result.values())
    assert sum(result.values()) == pytest.approx(1.0)


@pytest.mark.model
def test_classify_oversized_transcript():
    assert FINETUNED_MODEL_PATH.exists(), (
        f"Checkpoint not found at {FINETUNED_MODEL_PATH}. "
        "Run: python -m src.models.finetune_phrasebank"
    )
    scoring._load_finetuned.cache_clear()

    # ~3000 words -> several 510-token windows -> multi-chunk aggregation.
    long_text = (
        "The company delivered strong results this quarter with record revenue. "
    ) * 250

    result = scoring.classify_text(long_text)

    assert set(result) == set(LABELS)
    assert sum(result.values()) == pytest.approx(1.0)
