"""Live FinBERT scoring of ad-hoc transcript text — a thin composition layer.

Composes existing pure functions (no reimplementation, no ``mcp`` dependency):
    - src/data/chunk_transcripts.py :: chunk_speaker_text  (510-token windows)
    - src/models/inference.py       :: load_model, score_batch, aggregate_scores

Will chain chunk -> score -> mean-aggregate to turn raw speaker text into a
document-level sentiment result, loading the fine-tuned checkpoint at
``src/models/phrasebank_full_finetune/``.

Scaffolding step (ES-17/18): signature only, raising NotImplementedError.
:mod:`tools` does not call this yet — the stub step returns a hardcoded mock to
prove the FastMCP wiring; the real live-scoring path lands on
feature/classify-sentiment.
"""

# The real implementation composes the existing pure functions rather than
# reimplementing them, and imports the label contract via
# `from src.models.inference import LABELS` rather than redefining it:
#   src/data/chunk_transcripts.py :: chunk_speaker_text
#   src/models/inference.py       :: load_model, score_batch, aggregate_scores
# Chunk-to-document aggregation MUST match aggregate_scores (equal-weight mean
# of each class probability per document), the method used in ES-07/12.


def classify_text(transcript_text: str) -> dict[str, float]:
    """Score raw transcript text into document-level class probabilities.

    Cleans/chunks ``transcript_text`` (``chunk_speaker_text``, 510-token
    windows), scores each chunk with the fine-tuned checkpoint
    (``load_model`` + ``score_batch``), then aggregates chunk probabilities to
    one document result the same way the batch pipeline does
    (``aggregate_scores`` — equal-weight mean per class).

    Returns a ``{label: probability}`` mapping keyed by
    ``src.models.inference.LABELS``.

    TODO(feature/classify-sentiment): implement the chunk -> score -> aggregate
    composition against the fine-tuned checkpoint.
    """
    raise NotImplementedError("stub — see feature/classify-sentiment")
