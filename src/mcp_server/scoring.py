"""Live FinBERT scoring of ad-hoc transcript text — a thin composition layer.

Composes existing pure functions (no reimplementation, no ``mcp`` dependency):
    - src/data/chunk_transcripts.py :: chunk_speaker_text  (510-token windows)
    - src/models/inference.py       :: load_model, get_device, score_batch

:func:`classify_text` chains chunk -> score -> mean-aggregate to turn raw
speaker text into a document-level sentiment result, loading the fine-tuned
checkpoint at ``src/models/phrasebank_full_finetune/`` exactly once (eval mode)
via the cached :func:`_load_finetuned`.

Chunk-to-document aggregation is the equal-weight mean of each class probability
across chunks — identical to ``inference.aggregate_scores`` (the method used in
ES-07/12), computed here directly because the ad-hoc path has no
transcript/speaker/return invariants to carry through a DataFrame.
"""

from functools import lru_cache

from src.data.chunk_transcripts import chunk_speaker_text
from src.models.inference import (
    FINETUNED_MODEL_PATH,
    LABELS,
    get_device,
    load_model,
    score_batch,
)

# The fine-tuned checkpoint dir doubles as the model identity tag: no MLflow
# run_id is recorded into the checkpoint or scores anywhere in the pipeline.
MODEL_RUN_ID = FINETUNED_MODEL_PATH.name  # "phrasebank_full_finetune"


@lru_cache(maxsize=1)
def _load_finetuned():
    """Load the fine-tuned checkpoint once (eval mode) and cache it.

    Returns ``(tokenizer, model, device)``. ``load_model`` already calls
    ``model.eval()``; here it is moved onto the best available device. The
    ``lru_cache`` makes the ~439MB load happen a single time — on first classify
    or via :func:`warm_up` at server startup — and reused for every call.
    """
    tokenizer, model = load_model(str(FINETUNED_MODEL_PATH))
    device = get_device()
    model.to(device)
    return tokenizer, model, device


def warm_up() -> None:
    """Force the one-time checkpoint load (call at server startup)."""
    _load_finetuned()


def classify_text(transcript_text: str) -> dict[str, float]:
    """Score raw transcript text into document-level class probabilities.

    Chunks ``transcript_text`` into 510-token windows (``chunk_speaker_text``),
    scores each chunk with the fine-tuned checkpoint (``score_batch``), then
    aggregates chunk probabilities to one document result the same way the batch
    pipeline does (``aggregate_scores`` — equal-weight mean per class).

    Returns a ``{label: probability}`` mapping keyed by
    ``src.models.inference.LABELS`` (probabilities sum to 1).

    Raises ``ValueError`` when ``transcript_text`` yields no scoreable chunks
    (empty, whitespace-only, or shorter than the minimum window), since the
    result schema cannot represent an "unknown" sentiment.
    """
    tokenizer, model, device = _load_finetuned()
    chunks = chunk_speaker_text(transcript_text, tokenizer)  # 510-token windows
    if not chunks:
        raise ValueError(
            "transcript_text produced no scoreable chunks (empty or too short)"
        )
    probs = score_batch(chunks, tokenizer, model, device)  # (n_chunks, NUM_LABELS)
    doc = probs.mean(axis=0)  # equal-weight mean per class (see aggregate_scores)
    return {label: float(p) for label, p in zip(LABELS, doc)}
