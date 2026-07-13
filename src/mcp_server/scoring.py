"""Live FinBERT scoring of ad-hoc transcript text — a thin composition layer.

Composes existing pure functions (no reimplementation, no ``mcp`` dependency):
    - src/data/chunk_transcripts.py :: chunk_speaker_text  (510-token windows)
    - src/models/inference.py       :: load_model, score_batch, aggregate_scores

Will chain chunk -> score -> mean-aggregate to turn raw speaker text into a
document-level sentiment result, loading the fine-tuned checkpoint at
``src/models/phrasebank_full_finetune/``.

Scaffolding only — no implementation yet (ES-17/18).
"""
