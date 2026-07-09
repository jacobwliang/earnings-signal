# earnings-signal

Fine-tuned FinBERT pipeline that extracts sentiment from earnings call transcripts and measures correlation with short-term stock returns.

## Overview
- Downloads and preprocesses 500+ earnings call transcripts
- Runs baseline FinBERT sentiment scoring
- Fine-tunes FinBERT on Financial PhraseBank
- Correlates sentiment scores with 1d/3d/5d forward returns
- Tracks experiments with MLflow

## Stack
Python · HuggingFace Transformers · PyTorch · yfinance · Parquet · MLflow

## Quickstart
```bash
pip install -r requirements.txt
make pipeline
```

## Key findings

Earnings-call tone carries a **small but statistically reliable** signal about the next day's
stock move. Scoring CEO/CFO sentiment with the fine-tuned model and correlating against the
1-day market-adjusted (abnormal) return across **13,611 calls**:

- **Spearman rho = +0.10**, 95% CI [+0.087, +0.120] — sits fully above zero, so the relationship
  is very unlikely to be chance. Holds across 1d/5d windows and with/without market adjustment.
- **Not an artifact of the returns data.** An untrained-head control gives rho ≈ 0; the trained
  model gives +0.10. The difference (+0.106, p < 0.001) is the real proof the fine-tuning bought
  genuine signal.
- Validated at the **company level** (ticker-clustered bootstrap, 10k resamples), so the result
  isn't inflated by treating a single company's calls as independent.

**Takeaway:** positive calls tend to see better next-day performance, negative calls worse. The
effect is modest — best used as *one input among many*, not a standalone trading signal.

Full write-up: [reports/finetuned_correlation_analysis.md](reports/finetuned_correlation_analysis.md).

## Model performance

The fine-tuned FinBERT sentiment classifier (Financial PhraseBank) is strong and well-calibrated:

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Random-head baseline (chance floor) | ~random | — |
| Full fine-tune | **0.938** | **0.954** |

Light regularization (label smoothing, weight decay, dropout) keeps eval loss from drifting apart
from train loss, reducing overfitting at equal accuracy. Details:
[reports/baseline_analysis.md](reports/baseline_analysis.md),
[reports/full_finetune_run2_vs_run1.md](reports/full_finetune_run2_vs_run1.md).
