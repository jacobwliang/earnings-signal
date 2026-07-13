"""FastMCP server instantiation and tool registration; the process entry point.

Constructs ``mcp = FastMCP("earnings-signal")`` (official ``mcp`` SDK,
``mcp.server.fastmcp.FastMCP``), imports :mod:`tools` so the ``@mcp.tool()``
functions register against it, and exposes ``main(argv=None)`` that runs the
server over stdio transport.

Zero business logic lives here — it only assembles the server and starts it.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("earnings-signal")

# Import for side effects: registers the @mcp.tool() functions against `mcp`.
# Must follow the `mcp` binding above (tools.py does `from .server import mcp`),
# so this is import-ordering-safe, not a circular import.
from . import tools  # noqa: E402,F401


def main(argv: list[str] | None = None) -> None:
    """Run the MCP server over stdio transport (the process entry point)."""
    mcp.run()


if __name__ == "__main__":
    main()
