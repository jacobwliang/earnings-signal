"""Baseline FinBERT inference: score chunk texts and collapse to documents.

``score_batch`` is the core unit: it tokenizes a batch of chunk strings, runs a
forward pass under ``torch.no_grad()``, and returns softmax probabilities as a
plain NumPy array shaped ``(n_texts, n_classes)``.

``score_chunks`` runs ``score_batch`` over every row of ``chunks.parquet`` and
attaches the per-class probabilities; ``aggregate_scores`` then collapses those
chunk-level probabilities to one row per ``(transcript_id, speaker)`` — the
document-level ``baseline_scores.parquet`` consumed by the correlation analysis.

``get_device`` and ``load_model`` are thin loaders kept deliberately minimal;
the model is the pretrained FinBERT (``yiyanghkust/finbert-pretrain``) with a
3-class sequence-classification head. End-to-end scoring runs on Colab with a
CUDA GPU (``main`` / the importable functions).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_NAME = "yiyanghkust/finbert-pretrain"
NUM_LABELS = 3
MAX_LENGTH = 512
BATCH_SIZE = 32

# Paths resolve next to this script so it runs flat on Colab: drop
# chunks.parquet beside infer_baseline.py and baseline_scores.parquet is
# written back into the same folder.
HERE = Path(__file__).resolve().parent
CHUNKS_PATH = HERE / "chunks.parquet"
BASELINE_SCORES_PATH = HERE / "baseline_scores.parquet"

# FinBERT class index order; one named constant so it is trivial to flip.
LABELS = ("neutral", "positive", "negative")
PROB_COLUMNS = tuple(f"prob_{label}" for label in LABELS)

# Per-transcript invariants carried through aggregation unchanged.
_CARRY_COLUMNS = ("ticker", "return_start_date", "return_1d", "return_5d")

_SCORE_COLUMNS = (
    "transcript_id",
    "ticker",
    "return_start_date",
    "speaker",
    "n_chunks",
    *PROB_COLUMNS,
    "sentiment_score",
    "return_1d",
    "return_5d",
)


def get_device() -> torch.device:
    """Return the best available device: CUDA (Colab GPU), then MPS, then CPU.

    Inference is meant to run on a Colab CUDA GPU, but falling back to MPS
    (Apple Silicon) or CPU lets the same code run locally for smoke tests.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    """Load the FinBERT tokenizer and 3-class sequence-classification model."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    model.eval()
    return tokenizer, model


def score_batch(texts: list[str], tokenizer, model, device) -> np.ndarray:
    """Return softmax class probabilities for a batch of texts.

    Texts are tokenized with padding/truncation to ``MAX_LENGTH`` tokens and run
    through ``model`` on ``device`` under ``torch.no_grad()``. The logits are
    softmaxed over the class dimension and returned as a NumPy array of shape
    ``(len(texts), NUM_LABELS)``.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


def score_chunks(
    df: pd.DataFrame, tokenizer, model, device, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Attach per-chunk class probabilities to a copy of ``chunks.parquet``.

    Chunk texts are scored in ``batch_size`` slices via :func:`score_batch`
    (progress shown with ``tqdm``); the stacked ``(n, NUM_LABELS)`` probabilities
    are appended as :data:`PROB_COLUMNS`, preserving every original column
    (``transcript_id``, ``speaker``, returns, ...). An empty frame returns the
    same columns plus the prob columns with no rows.
    """
    out = df.copy()
    texts = out["chunk_text"].tolist()

    if not texts:
        for col in PROB_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    batches = [
        score_batch(texts[start:start + batch_size], tokenizer, model, device)
        for start in tqdm(
            range(0, len(texts), batch_size), desc="scoring chunks"
        )
    ]
    probs = np.vstack(batches)
    out[list(PROB_COLUMNS)] = probs
    return out


def aggregate_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse chunk-level scores to one row per ``(transcript_id, speaker)``.

    Grouping on both keys is the CEO/CFO independence guarantee: each speaker's
    chunks are averaged on their own and never mixed with the other speaker or
    with another transcript. Each class probability is an equal-weight mean over
    the group's chunks; ``n_chunks`` is the group size; per-transcript invariants
    (ticker, dates, returns) are carried through unchanged. ``sentiment_score``
    is ``mean P(positive) - mean P(negative)``.
    """
    if scored.empty:
        return pd.DataFrame(columns=list(_SCORE_COLUMNS))

    agg = {col: "mean" for col in PROB_COLUMNS}
    agg.update({col: "first" for col in _CARRY_COLUMNS})

    out = (
        scored.groupby(["transcript_id", "speaker"], sort=True)
        .agg(**{col: (col, how) for col, how in agg.items()},
             n_chunks=("chunk_text", "size"))
        .reset_index()
    )
    out["sentiment_score"] = out["prob_positive"] - out["prob_negative"]
    return out[list(_SCORE_COLUMNS)]


def log_stats(out: pd.DataFrame) -> None:
    """Print total rows, per-speaker counts, and sentiment_score mean/std."""
    print(f"Total rows: {len(out)}")
    print("Rows by speaker:")
    for speaker, count in out["speaker"].value_counts().items():
        print(f"  {speaker}: {count}")
    print(
        f"sentiment_score: mean={out['sentiment_score'].mean():.4f}  "
        f"std={out['sentiment_score'].std():.4f}"
    )


def main() -> None:
    """Score chunks.parquet and write document-level baseline_scores.parquet."""
    print(f"Loading model {MODEL_NAME} ...")
    tokenizer, model = load_model()
    device = get_device()
    model.to(device)
    print(f"Model loaded on device: {device}")

    print(f"Reading chunks from {CHUNKS_PATH} ...")
    chunks = pd.read_parquet(CHUNKS_PATH)
    print(f"Read {len(chunks)} chunks; scoring in batches of {BATCH_SIZE} ...")

    scored = score_chunks(chunks, tokenizer, model, device)
    print("Scoring done; aggregating to document level ...")
    out = aggregate_scores(scored)

    log_stats(out)
    out.to_parquet(BASELINE_SCORES_PATH, index=False)
    print(f"Wrote {len(out)} rows to {BASELINE_SCORES_PATH}")


if __name__ == "__main__":
    main()
