"""Parquet load/filter helpers over the project's canonical data locations.

Plain functions returning ``pandas.DataFrame`` — no ``mcp`` dependency, so this
module stays pure and unit-testable with inline DataFrames.

NOTE: ``src/models/inference.py`` defines Colab-flat default paths (e.g.
``CHUNKS_PATH = HERE / "chunks.parquet"``) that do NOT point at the canonical
locations. This module deliberately defines its own correct path constants and
callers pass explicit paths rather than relying on those module defaults.

Canonical locations to expose (constants to be added at implementation):
    data/processed/chunks.parquet
    data/processed/master_clean.parquet
    results/inference/baseline_scores.parquet
    results/inference/finetuned_scores.parquet
    src/models/phrasebank_full_finetune/   (fine-tuned checkpoint dir)

Scaffolding only — no constants or functions defined yet (ES-17/18).
"""
