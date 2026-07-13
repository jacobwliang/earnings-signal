"""Unit tests for src/mcp_server/data_access.py.

House style: build inputs as inline pandas DataFrames (no file I/O), assert
structural invariants exactly. Reads of the real gitignored parquet files, if
any, are gated behind ``@pytest.mark.data`` (excluded in CI via ``-m "not data"``).

Scaffolding only — no tests yet (ES-17/18).
"""
