"""FastMCP server instantiation and tool registration; the process entry point.

Will construct ``mcp = FastMCP("earnings-signal")`` (official ``mcp`` SDK,
``mcp.server.fastmcp.FastMCP``), import :mod:`tools` so the ``@mcp.tool()``
functions register against it, and expose ``main(argv=None)`` that runs the
server over stdio transport.

Zero business logic lives here — it only assembles the server and starts it.

Scaffolding only — no implementation yet (ES-17/18).
"""
