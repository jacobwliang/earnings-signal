# earnings-signal

**Fine-tuning FinBERT on earnings call transcripts to predict short-term stock returns — with the entire pipeline organized around preventing one class of error: lookahead bias.**
https://github.com/jacobwliang/earnings-signal

---

## Overview

- End-to-end NLP/ML pipeline: earnings call transcripts → fine-tuned financial sentiment model → correlation with short-term stock returns
- Central design constraint: **zero lookahead bias** — no feature or label may be derived from information not available at prediction time. Shapes price windowing, split construction, and label timing throughout.
- **Stack:** Python, HuggingFace Transformers, PyTorch, yfinance, Parquet, MLflow, GitHub Actions
- Inference run on Google Colab (T4 GPU)

---

## System design

### Pipeline

| Stage | Description |
|---|---|
| ES-01/02 | Transcript collection |
| ES-03 | Price data + return calculation via yfinance |
| ES-04 | Join into a master dataset |
| ES-05 | Transcript cleaning / speaker section splitting |
| ES-06 | Chunking to fit the 512-token BERT limit |
| ES-07 | Baseline inference (untrained classification head) |
| ES-08 | Correlation analysis — baseline |
| ES-09/10 | Financial PhraseBank fine-tuning |
| ES-11 | MLflow experiment tracking |
| ES-12 | Fine-tuned correlation + comparison vs. baseline |
| ES-13 | Subgroup analysis (market cap, COVID-era stress test) |
| ES-14 | Visualizations |
| ES-15 | Limitations documentation |
| ES-16 | README |
| ES-17/18 | MCP server integration (stretch) |
Note: ES just stands for earnings signal

### Data flow and shape

- 17,540 raw transcripts → ~13,600 fully labeled rows after price-join filtering (ES-03/04)
- ES-05 splits each transcript into CEO, CFO, and "other executive" remarks — 87% coverage where both CEO and CFO sections parse cleanly
- ES-06 chunks each speaker's remarks into non-overlapping 510-token windows — ~141,000 chunks total, ~8 per transcript on average, dense enough that non-overlapping windows don't meaningfully lose boundary signal
- `transcript_id = ticker + "_" + return_start_date`

### Baseline

- Backbone: `yiyanghkust/finbert-pretrain` (financial-domain pretrained, no classification head) + randomly initialized classification head
- Deliberate chance-floor baseline, not a strawman — establishes what "no signal" looks like before fine-tuning
- Results: 25,619 document-speaker rows across 140,196 chunks; correlation with 1-day returns of −0.001 (expected for an untrained head); 93% positive-argmax skew reflecting an untrained decision boundary, not genuine sentiment
- `ProsusAI/finbert` (already fine-tuned on Financial PhraseBank) is deliberately excluded as a baseline — using it would contaminate the later fine-tuning comparison

### Fine-tuning (current stage)

- Dataset: Financial PhraseBank, `sentences_75agree` config, 3,453 sentences — 62.1% neutral, 25.7% positive, 12.2% negative
- Each sentiment name is encoded at load time to the project-standard label ids `{0: neutral, 1: positive, 2: negative}` (matches `src/models/inference.py`; the index is the label id the fine-tuned head learns)
- Split: 80/10/10, stratified, seed 42; proportion-drift checks (2% tolerance) and index-overlap checks; persisted to Parquet for reproducibility
- Evaluation is a baseline ladder, not a single before/after: chance-floor baseline → linear probe (frozen backbone + logistic regression) → full fine-tune, with `finbert-tone` as an off-the-shelf reference point. The linear probe isolates fine-tuning's contribution from the backbone's pretraining.
- Caveat: test split has only ~42 negative-class examples → ±14pp confidence interval on negative-class recall; treated as a real evaluation limitation, not a footnote

### Correlation analysis (upcoming)

- ES-08 and ES-12 carry the most statistical validity risk: multiple testing across return windows/speaker sources/correlation methods, underpowered correlations, and effect size vs. significance
- Treated as requiring more methodological care than the modeling stages

### Known limitation

- ~20.7% of transcript rows affected by ticker download failures (post-retry), concentrated in the 2021 SPAC/IPO cohort and acquired/delisted companies
- Documented as a survivorship bias limitation (ES-15), not patched

---

## Engineering conventions

- Modular `src/` layout, docstrings on all functions
- Unit tests use inline `pd.DataFrame` construction, no file I/O; integration tests marked separately and excluded from CI
- Parquet for all persisted data, including frozen train/val/test splits
- Branching: `feature/`, `fix/`, `setup/` prefixes; atomic commits referencing issues