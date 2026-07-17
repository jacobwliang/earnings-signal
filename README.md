# earnings-signal

Fine-tuned FinBERT pipeline that extracts sentiment from earnings call transcripts and measures correlation with short-term stock returns.

## Overview
- Downloads and preprocesses 500+ earnings call transcripts
- Runs baseline FinBERT sentiment scoring
- Fine-tunes FinBERT on Financial PhraseBank
- Correlates sentiment scores with 1d/3d/5d forward returns

## Stack
Python · HuggingFace Transformers · PyTorch · yfinance · Parquet · MLflow

## Quickstart
```bash
pip install -r requirements.txt
make pipeline
```

## MCP Server
An [MCP](https://modelcontextprotocol.io) server exposes the pipeline as tools for Claude and other MCP clients:
- `list_covered_tickers` — discover which tickers/calls exist (counts, date range) with optional prefix/limit
- `get_ticker_sentiment_history` — persisted per-call sentiment for a ticker over an optional date range
- `compare_tickers` — most recent classification across several tickers
- `get_transcript` — the stored prepared transcript text for one call, by speaker section
- `classify_earnings_sentiment` — score arbitrary transcript text live against the fine-tuned checkpoint

Registered via [.mcp.json](.mcp.json); run with `.venv/bin/python -m src.mcp_server`. See [reports/mcp_server.md](reports/mcp_server.md) for details.

## Results
_To be filled in after experiments._
