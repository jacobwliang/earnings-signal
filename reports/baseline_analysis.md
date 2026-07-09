# Baseline Inference Analysis

## Purpose

`yiyanghkust/finbert-pretrain` ran with a **randomly-initialized 3-class classification head** 
over the transcript chunks, producing per-chunk and document-level positive/negative/neutral probabilities.

This is the **pre-fine-tuning baseline**, the lower bound the project
measures everything else against. It is not expected to carry sentiment signal.

---

## Why a random head

- `yiyanghkust/finbert-tone` are both already fine-tuned on
  sentiment/tone tasks. Using either as the baseline would **contaminate** the
  ES-09/10 comparison, since that stage also fine-tunes on sentiment data
  (Financial PhraseBank). The comparison would then be "model that already saw
  sentiment training" vs "model that saw sentiment training" — isolating nothing.
- A randomly-initialized head gives the honest answer to "what does the
  domain-adapted backbone yield with *zero* sentiment supervision?"
- ES-07's deliverable is therefore the **shared inference machinery** plus a
  **documented chance floor** — not scores that correlate with returns. The real
  signal comes from fine-tuning this same backbone in ES-09/10.

The label order `{0: neutral, 1: positive, 2: negative}` (the canonical
`finbert-tone` order) is fixed here, because it is the contract ES-09/10
fine-tuning depends on.

---

## What was run

- **Backbone:** `yiyanghkust/finbert-pretrain` (domain-adapted BERT, no task head)
- **Head:** 3-class sequence-classification head, randomly initialized
- **Environment:** Google Colab (T4 GPU)
- **Coverage:** 140,196 chunks → 25,619 document-speaker rows
  (13,093 CEO + 12,526 CFO), across 13,611 transcripts and 2,172 tickers,
  spanning 2017-11 → 2023-02.
- **Invariants verified:** probabilities sum to 1; `sentiment_score = prob_pos −
  prob_neg`; per-group chunk counts reconcile to all 140,196 source chunks; no
  duplicate groups; no structural nulls (except inherited return nulls, below).

---

## Results — flat by construction (the intended chance floor)

**The scores barely move.**

- Three probabilities per transcript-speaker, summing to 1. Across all 25,619
  rows: mean ≈ 0.37 / 0.34 / 0.29 (pos / neg / neu), std ≈ 0.014.
- The model gave nearly the same answer for every call — a blowout quarter and a
  disaster look almost identical. That flatness is the point.
- A *trained* model would be confident, often 0.7–0.95 on the top class. Ours
  never gets there: 98% of rows have a top probability under 0.40. It sits at "I
  have no idea," which is correct for an untrained head — it is guessing, not
  reading.

| metric | value |
|---|---|
| mean prob_pos / neg / neu | 0.373 / 0.336 / 0.291 |
| std of each prob | ~0.014 |
| sentiment_score (pos − neg) | +0.038 (sd 0.025) |
| range | −0.09 to +0.18 |

- corr(sentiment, return_1d) = −0.001 (n=25,269)
- corr(sentiment, return_5d) = −0.007 (n=25,424)
- Both are indistinguishable from zero. There is no relationship, because there is no real signal in the scores to begin with.

---

## Known limitations

- **No sentiment signal by design** — scores are ~1/3 each; do not interpret them
  as tone.
- **Missing forward returns** — `return_1d` has 350 nulls, `return_5d` 195
  (inherited from source chunks without price data). ES-08 must `dropna` on the
  return column before correlating, or the join will silently shrink.
- **Thin documents** — 888 groups are single-chunk (median `n_chunks` = 4, max
  47). Single-chunk speakers have higher-variance scores; an `n_chunks ≥ 2`
  sensitivity check is deferred until a trained head exists.

---

## Reproduce

```bash
python -m src.models.inference \
    --input data/chunks.parquet \
    --output data/baseline_scores.parquet \
    --batch-size 64
```

Raw output Parquets are intentionally not committed (140k+ rows); regenerate with
the command above. Heavy inference was run on Colab (T4); the repo code is the
source of truth and runs identically with the `cuda → mps → cpu` device fallback.

---

## Outputs

- `chunks_scored.parquet` — per chunk, adds `prob_neutral / prob_positive / prob_negative`
- `baseline_scores.parquet` — per `(transcript_id, speaker)`