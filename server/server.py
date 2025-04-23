"""
Browser Use MCP Server

This module implements an MCP (Model-Control-Protocol) server for browser automation
using the browser_use library. It provides functionality to interact with a browser instance
via an async task queue, allowing for long-running browser tasks to be executed asynchronously
while providing status updates and results.

The server supports Server-Sent Events (SSE) for web-based interfaces.
"""

# Standard library imports
import os
import asyncio
import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
import time
import sys

# Third-party imports
import click
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger

# Browser-use library imports
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# MCP server components
from mcp.server import Server
import mcp.types as types

# LLM provider
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logger = logging.getLogger()
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler(sys.stderr)
formatter = jsonlogger.JsonFormatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ensure uvicorn also logs to stderr in JSON format
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []
uvicorn_logger.addHandler(handler)

# Ensure all other loggers use the same format
logging.getLogger("browser_use").addHandler(handler)
logging.getLogger("playwright").addHandler(handler)
logging.getLogger("mcp").addHandler(handler)

# Load environment variables
load_dotenv()


def parse_bool_env(env_var: str, default: bool = False) -> bool:
    """
    Parse a boolean environment variable.

    Args:
        env_var: The environment variable name
        default: Default value if not set

    Returns:
        Boolean value of the environment variable
    """
    value = os.environ.get(env_var)
    if value is None:
        return default

    # Consider various representations of boolean values
    return value.lower() in ("true", "yes", "1", "y", "on")


def init_configuration() -> Dict[str, Any]:
    """
    Initialize configuration from environment variables with defaults.

    Returns:
        Dictionary containing all configuration parameters
    """
    config = {
        # Browser window settings
        "DEFAULT_WINDOW_WIDTH": int(os.environ.get("BROWSER_WINDOW_WIDTH", 1280)),
        "DEFAULT_WINDOW_HEIGHT": int(os.environ.get("BROWSER_WINDOW_HEIGHT", 1100)),
        # Browser config settings
        "DEFAULT_LOCALE": os.environ.get("BROWSER_LOCALE", "en-US"),
        "DEFAULT_USER_AGENT": os.environ.get(
            "BROWSER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
        ),
        # Task settings
        "MAX_AGENT_STEPS": int(os.environ.get("MAX_AGENT_STEPS", 10)),
        # Browser arguments
        "BROWSER_ARGS": [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",  # Use random port to avoid conflicts
        ],
    }

    return config


# Initialize configuration
CONFIG = init_configuration()

def create_mcp_server(
    window_width: int = CONFIG["DEFAULT_WINDOW_WIDTH"],
    window_height: int = CONFIG["DEFAULT_WINDOW_HEIGHT"],
    locale: str = CONFIG["DEFAULT_LOCALE"],
) -> Server:
    """
    Create and configure an MCP server for browser interaction.

    Args:
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale

    Returns:
        Configured MCP server instance
    """
    # Create MCP server instance
    app = Server("browser_use")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
        """
        Handle tool calls from the MCP client.

        Args:
            name: The name of the tool to call
            arguments: The arguments to pass to the tool

        Returns:
            A list of content objects containing the result of the browser task.

        Raises:
            ValueError: If required arguments are missing or the tool is unknown.
        """
        if name == "run_browser_task":
            if "task" not in arguments:
                raise ValueError("Missing required argument 'task'")

            task_desc = arguments["task"]
            browser = None
            context = None
            task_result_data = {"status": "pending"}

            try:
                # Get Chrome path from environment if available                
                
                browser = Browser(
                    config=BrowserConfig(
                        # Specify the path to your Chrome executable
                        chrome_instance_path=os.getenv('CHROME_PATH',"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),  # macOS path
                        
                        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
                        # For Linux, typically: '/usr/bin/google-chrome'
                    )
                )

                # Configure LLM based on environment variables
                model_name = os.getenv('OPENAI_MODEL', "gpt-4o-mini")
                try:
                    temperature_str = os.getenv('OPENAI_TEMPERATURE', '0.0')
                    temperature = float(temperature_str)
                except ValueError:
                    logger.warning(f"Could not convert OPENAI_TEMPERATURE value '{temperature_str}' to float. Using default 0.0.")
                    temperature = 0.0

                api_key = os.getenv('OPENAI_API_KEY')
                if api_key is None:
                    logger.error("OPENAI_API_KEY environment variable is not set.")
                    raise ValueError("Error: OPENAI_API_KEY environment variable is not set.")

                openai_api_base = os.getenv('OPENAI_API_BASE',"https://api.openai.com/v1")

                llm = ChatOpenAI(
                    model=model_name,              # Use variable
                    temperature=temperature,       # Use variable
                    api_key=api_key,               # Use variable
                    openai_api_base=openai_api_base # Use variable
                )

                # Create agent with the fresh context and configured LLM
                agent = Agent(
                    task=task_desc,
                    llm=llm,
                    browser=browser,
                )

                # Run the agent with a step limit
                logger.info(f"Starting browser task: {task_desc}")
                agent_result = await agent.run(max_steps=CONFIG["MAX_AGENT_STEPS"])
                logger.info(f"Browser task finished.")

                # Get the final result
                final_result = agent_result.final_result()

                if final_result and hasattr(final_result, "raise_for_status"):
                    final_result.raise_for_status()
                    result_text = str(final_result.text)
                else:
                    result_text = str(final_result) if final_result else "No final result available"

                # Gather essential information
                is_successful = agent_result.is_successful()
                has_errors = agent_result.has_errors()
                errors = agent_result.errors()
                urls_visited = agent_result.urls()
                action_names = agent_result.action_names()
                extracted_content = agent_result.extracted_content()
                steps_taken = agent_result.number_of_steps()

                # Create focused response
                task_result_data = {
                    "status": "completed" if is_successful else "failed",
                    "final_result": result_text,
                    "success": is_successful,
                    "has_errors": has_errors,
                    "errors": [str(err) for err in errors if err],
                    "urls_visited": [str(url) for url in urls_visited if url],
                    "actions_performed": action_names,
                    "extracted_content": extracted_content,
                    "steps_taken": steps_taken,
                }

            except Exception as e:
                logger.error(f"Error running browser task '{task_desc}': {str(e)}")
                tb = traceback.format_exc()
                task_result_data = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": tb,
                }
            finally:
                # Clean up browser resources
                try:
                    if context:
                        await context.close()
                    if browser:
                        await browser.close()
                    logger.info(f"Browser resources cleaned up for task: {task_desc[:50]}...")
                except Exception as e:
                    logger.error(f"Error cleaning up browser resources: {str(e)}")

            # Return the result
            return [types.TextContent(type="text", text=json.dumps(task_result_data, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """
        List the available tools for the MCP client.

        Returns:
            A list containing the single 'run_browser_task' tool definition.
        """
        return [
            types.Tool(
                name="run_browser_task",
                description="Runs a browser automation task using the Agent.",
                inputSchema={
                    "type": "object",
                    "required": ["task"],
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "A detailed description of the task to perform in the browser.",
                        },
                    },
                },
            )
        ]

    return app


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--proxy-port",
    default=None,
    type=int,
    help="Port for the proxy to listen on. If specified, enables proxy mode.",
)
@click.option("--chrome-path", default=None, help="Path to Chrome executable")
@click.option(
    "--window-width",
    default=CONFIG["DEFAULT_WINDOW_WIDTH"],
    help="Browser window width",
)
@click.option(
    "--window-height",
    default=CONFIG["DEFAULT_WINDOW_HEIGHT"],
    help="Browser window height",
)
@click.option("--locale", default=CONFIG["DEFAULT_LOCALE"], help="Browser locale")
@click.option(
    "--stdio",
    is_flag=True,
    default=False,
    help="Enable stdio mode. If specified, enables proxy mode.",
)
def main(
    port: int,
    proxy_port: Optional[int],
    chrome_path: str,
    window_width: int,
    window_height: int,
    locale: str,
    stdio: bool,
) -> int:
    """
    Run the browser-use MCP server.

    This function initializes the MCP server and runs it with the SSE transport.
    Browser tasks are executed synchronously within the tool call.

    The server can run in two modes:
    1. Direct SSE mode (default): Just runs the SSE server
    2. Proxy mode (enabled by --stdio or --proxy-port): Runs both SSE server and mcp-proxy

    Args:
        port: Port to listen on for SSE
        proxy_port: Port for the proxy to listen on. If specified, enables proxy mode.
        chrome_path: Path to Chrome executable
        window_width: Browser window width
        window_height: Browser window height
        locale: Browser locale
        stdio: Enable stdio mode. If specified, enables proxy mode.

    Returns:
        Exit code (0 for success)
    """
    # Store Chrome path in environment variable if provided
    if chrome_path:
        os.environ["CHROME_PATH"] = chrome_path
        logger.info(f"Using Chrome path: {chrome_path}")
    else:
        # Check if CHROME_PATH is already set in environment
        env_chrome_path = os.environ.get("CHROME_PATH")
        if env_chrome_path:
             logger.info(f"Using Chrome path from environment: {env_chrome_path}")
        else:
            logger.info(
                "No Chrome path specified via argument or environment variable, letting Playwright use its default browser"
            )

    # Create MCP server
    app = create_mcp_server(
        window_width=window_width,
        window_height=window_height,
        locale=locale,
    )

    # Set up SSE transport
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn
    import asyncio
    import threading

    sse = SseServerTransport("/messages/")

    # Create the Starlette app for SSE
    async def handle_sse(request):
        """Handle SSE connections from clients."""
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Error in handle_sse: {str(e)}")
            raise

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    # Add startup event
    @starlette_app.on_event("startup")
    async def startup_event():
        """Initialize the server on startup."""
        logger.info("Starting MCP server...")

        # Sanity checks for critical configuration
        if port <= 0 or port > 65535:
            logger.error(f"Invalid port number: {port}")
            raise ValueError(f"Invalid port number: {port}")

        if window_width <= 0 or window_height <= 0:
            logger.error(f"Invalid window dimensions: {window_width}x{window_height}")
            raise ValueError(
                f"Invalid window dimensions: {window_width}x{window_height}"
            )

    # Function to run uvicorn in a separate thread
    def run_uvicorn():
        # Configure uvicorn to use JSON logging
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": "INFO"},
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO"},
            },
        }

        uvicorn.run(
            starlette_app,
            host="0.0.0.0",
            port=port,
            log_config=log_config,
            log_level="info",
        )

    # If proxy mode is enabled, run both the SSE server and mcp-proxy
    if stdio:
        import subprocess

        # Start the SSE server in a separate thread
        sse_thread = threading.Thread(target=run_uvicorn)
        sse_thread.daemon = True
        sse_thread.start()

        # Give the SSE server a moment to start
        time.sleep(1)

        proxy_cmd = [
            "mcp-proxy",
            f"http://localhost:{port}/sse",
            "--sse-port",
            str(proxy_port),
            "--allow-origin",
            "*",
        ]

        logger.info(f"Running proxy command: {' '.join(proxy_cmd)}")
        logger.info(
            f"SSE server running on port {port}, proxy running on port {proxy_port}"
        )

        try:
            with subprocess.Popen(proxy_cmd) as proxy_process:
                proxy_process.wait()
        except Exception as e:
            logger.error(f"Error starting mcp-proxy: {str(e)}")
            logger.error(f"Command was: {' '.join(proxy_cmd)}")
            return 1
    else:
        logger.info(f"Running in direct SSE mode on port {port}")
        run_uvicorn()

    return 0


if __name__ == "__main__":
    main()
