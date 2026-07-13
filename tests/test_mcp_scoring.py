"""Unit tests for src/mcp_server/scoring.py.

Fast tests use a hand-written fake tokenizer/model stand-in (see
``WhitespaceTokenizer`` in tests/test_finetune_phrasebank.py) rather than
unittest.mock, so no checkpoint download or forward pass is needed. Any test
that loads the real FinBERT checkpoint is gated behind a ``model`` marker
(excluded in CI).

Scaffolding only — no tests yet (ES-17/18).
"""
