# MCP Server

An [MCP](https://modelcontextprotocol.io) server that exposes the earnings-call
sentiment pipeline as tools for Claude (and any MCP client). Assistants can discover
coverage, look up per-call sentiment, compare tickers, read stored transcript text,
and score raw transcript text against the fine-tuned FinBERT checkpoint — without
touching parquet files or model code.

## Tools

- **`list_covered_tickers(prefix?, limit?)`** — discovery for "what do you have?": the
  covered tickers (with per-ticker call counts), overall earnings-date range, and total
  call count. `prefix` (case-insensitive) and `limit` keep the list small; a separate
  `covered_ticker_count` reports the full match count so scope stays visible when the
  list is capped.
- **`get_ticker_sentiment_history(ticker, start_date?, end_date?)`** — persisted sentiment for
  a ticker over an optional ISO date range. CEO + CFO rows are averaged into one row
  per call. A `ticker_covered` flag separates two empty cases: never processed
  (`False`) vs. covered but no calls in range (`True`, `match_count=0`).
- **`compare_tickers(tickers)`** — most recent classification per ticker (`None` when
  covered-but-empty or not covered). Case-insensitive duplicates collapse, keeping
  first-seen order.
- **`get_transcript(ticker, earnings_date, speaker?)`** — the stored `master_clean`
  prepared text for one call, joined on `(ticker, return_start_date)`. `speaker`
  (`ceo`/`cfo`/`other_exec`) returns a single section; omitting it returns all
  available ones. Sections run long, so the combined payload is capped (50k chars)
  with a `truncated` flag so callers know when they aren't seeing the full text.
  Raises `ValueError` for an unknown speaker, malformed date, or no matching call.
- **`classify_earnings_sentiment(transcript_text)`** — runs FinBERT **live** over
  arbitrary text, reusing the pipeline's cleaning/chunking (510-token windows) and
  equal-weight aggregation so results match the batch pipeline. Raises `ValueError`
  if no scoreable chunks.

All outputs are sentiment classifications correlated with the transcript's language
— **not** return predictions or forecasts.

## Architecture

Business logic stays in pure, testable modules; MCP wiring stays thin. Crucially,
`data_access` and `scoring` have **no MCP dependency**, so they unit-test with inline
DataFrames and text.

| Module | Role |
| --- | --- |
| [`__main__.py`](../src/mcp_server/__main__.py) | Entry point for `python -m src.mcp_server`. |
| [`server.py`](../src/mcp_server/server.py) | Builds `FastMCP`, registers tools, warms up the model, runs over stdio. |
| [`tools.py`](../src/mcp_server/tools.py) | The five `@mcp.tool()` wrappers — validate/serialize only. |
| [`data_access.py`](../src/mcp_server/data_access.py) | Pandas load/filter helpers over the parquet locations. |
| [`scoring.py`](../src/mcp_server/scoring.py) | Live FinBERT scoring of ad-hoc text. |
| [`schemas.py`](../src/mcp_server/schemas.py) | Pydantic v2 result models; tool output schemas derive from these. |

## Implementation notes

- **Canonical paths.** `data_access` defines correct path constants
  (`data/processed/`, `results/inference/`) instead of inheriting the Colab-flat
  defaults in `src/models/inference.py`.
- **Warm-up at startup.** `server.main` loads the ~439MB checkpoint once up front
  (via `lru_cache`) so the first live classify call isn't slow.
- **Model identity.** No MLflow `run_id` is stored, so the checkpoint dir name
  (`phrasebank_full_finetune`) doubles as `MODEL_RUN_ID`.
- **Mock seams.** `_read_scores` and `_read_master_clean` are the parquet-reading
  seams, monkeypatched in tests so no files are touched.

## Testing & config

- Tests: [`test_mcp_data_access.py`](../tests/test_mcp_data_access.py),
  [`test_mcp_scoring.py`](../tests/test_mcp_scoring.py),
  [`test_mcp_tools.py`](../tests/test_mcp_tools.py) — each pure layer with inline data.
- A `model` pytest marker ([pytest.ini](../pytest.ini)) flags tests needing the real
  checkpoint. CI runs `-m "not data and not model"` to skip dataset/checkpoint tests.
- `mcp==1.28.1` in [requirements.txt](../requirements.txt);
  [.mcp.json](../.mcp.json) launches `.venv/bin/python -m src.mcp_server`.
