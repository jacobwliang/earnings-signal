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

## Results
_To be filled in after experiments._
