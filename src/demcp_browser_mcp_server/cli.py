"""
Command line interface for demcp_browser_mcp.

This module provides a command-line interface for starting the demcp_browser_mcp server.
It wraps the existing server functionality with a CLI.
"""

import os
import sys
import json
import logging
import click
import importlib.util
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler(sys.stderr)
formatter = jsonlogger.JsonFormatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_error(message: str, error: Exception = None):
    """Log error in JSON format to stderr"""
    error_data = {"error": message, "traceback": str(error) if error else None}
    print(json.dumps(error_data), file=sys.stderr)


def import_server_module():
    """
    Import the server module from the server directory.
    This allows us to reuse the existing server code.
    """
    # Add the root directory to the Python path to find server module
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root_dir)

    try:
        # Try to import the server module
        import server.server

        return server.server
    except ImportError:
        # If running as an installed package, the server module might be elsewhere
        try:
            # Look in common locations
            if os.path.exists(os.path.join(root_dir, "server", "server.py")):
                spec = importlib.util.spec_from_file_location(
                    "server.server", os.path.join(root_dir, "server", "server.py")
                )
                server_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(server_module)
                return server_module
        except Exception as e:
            log_error("Could not import server module", e)
            raise ImportError(f"Could not import server module: {e}")

        raise ImportError(
            "Could not find server module. Make sure it's installed correctly."
        )


@click.group()
def cli():
    """Browser-use MCP server command line interface."""
    pass


@cli.command()
@click.argument("subcommand")
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--proxy-port",
    default=None,
    type=int,
    help="Port for the proxy to listen on (when using stdio mode)",
)
@click.option("--chrome-path", default=None, help="Path to Chrome executable")
@click.option("--window-width", default=1280, help="Browser window width")
@click.option("--window-height", default=1100, help="Browser window height")
@click.option("--locale", default="en-US", help="Browser locale")
# @click.option(
#     "--task-expiry-minutes",
#     default=60,
#     help="Minutes after which tasks are considered expired",
# )
@click.option(
    "--stdio", is_flag=True, default=False, help="Enable stdio mode with mcp-proxy"
)
def run(
    subcommand,
    port,
    proxy_port,
    chrome_path,
    window_width,
    window_height,
    locale,
    stdio,
):
    """Run the browser-use MCP server.

    SUBCOMMAND: should be 'server'
    """
    if subcommand != "server":
        log_error(f"Unknown subcommand: {subcommand}. Only 'server' is supported.")
        sys.exit(1)

    try:
        # Import the server module
        server_module = import_server_module()

        # We need to construct the command line arguments to pass to the server's Click command
        old_argv = sys.argv.copy()

        # Build a new argument list for the server command
        new_argv = [
            "server",  # Program name
            "--port",
            str(port),
        ]

        if chrome_path:
            new_argv.extend(["--chrome-path", chrome_path])

        if proxy_port is not None:
            new_argv.extend(["--proxy-port", str(proxy_port)])

        new_argv.extend(["--window-width", str(window_width)])
        new_argv.extend(["--window-height", str(window_height)])
        new_argv.extend(["--locale", locale])

        if stdio:
            new_argv.append("--stdio")

        # Replace sys.argv temporarily
        sys.argv = new_argv

        # Run the server's command directly
        try:
            return server_module.main()
        finally:
            # Restore original sys.argv
            sys.argv = old_argv

    except Exception as e:
        log_error("Error starting server", e)
        sys.exit(1)


if __name__ == "__main__":
    cli()
