"""``@mcp.tool()``-decorated functions exposed by the MCP server.

Thin wrappers only: each tool validates/serializes and delegates to
:mod:`data_access` (parquet reads/filters) or :mod:`scoring` (live FinBERT
scoring). No business logic inline.

The concrete tool surface is intentionally undecided at this scaffolding step —
it will be filled in once the tools to expose are specified.

Scaffolding only — no tools implemented yet (ES-17/18).
"""
