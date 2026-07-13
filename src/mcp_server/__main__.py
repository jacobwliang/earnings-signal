"""Entry point for ``python -m src.mcp_server``.

Delegates to ``server.main`` so the server launches via the repo's standard
``python -m src.<pkg>.<module>`` convention (matching the other runnable modules
under ``src/``). ``.mcp.json`` registers this module with Claude Code.

Scaffolding only — wiring to be implemented in ES-17/18.
"""
