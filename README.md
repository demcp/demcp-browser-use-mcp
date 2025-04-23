# demcp_browser_mcp

<div align="center">

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/DeMCP_AI.svg?style=social&label=Follow%20%40DeMCP_AI)](https://x.com/DeMCP_AI)
[![PyPI version](https://badge.fury.io/py/demcp_browser_mcp.svg)](https://pypi.org/project/demcp_browser_mcp/)

**An MCP server that enables AI agents to control web browsers using
[browser-use](https://github.com/browser-use/browser-use).**

</div>

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- [Playwright](https://playwright.dev/) - Browser automation
- [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) - Required for stdio mode
- [browser-use-mcp-server](https://github.com/co-browser/browser-use-mcp-server) - browser-use mcp server

```bash
# Install prerequisites manually (Example for macOS using Homebrew)
brew install python@3.11 # Ensure Python 3.11+ is installed
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install mcp-proxy
# Ensure uv's bin directory is in your PATH (e.g., ~/.cargo/bin)
```

## Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key # Required
CHROME_PATH=optional/path/to/chrome # Optional, if not in standard location
OPENAI_MODEL=gpt-4o # Optional, defaults to gpt-4o-mini in server code
# ... other optional env vars
```

## Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd demcp_browser_mcp

# Create virtual environment (recommended)
uv venv
source .venv/bin/activate # On Linux/macOS
# .\.venv\Scripts\Activate.ps1 # On Windows PowerShell

# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install --with-deps --no-shell chromium
```

## Automated Setup from Scratch (Using Scripts)

For a fresh machine setup, you can use the provided scripts to automate the installation of prerequisites and project setup.

**Note:** These scripts require user interaction (e.g., entering API keys, confirming installations, entering sudo passwords) and might need terminal restarts afterwards for PATH changes to take effect.

1.  **Download the appropriate script** for your operating system (`start.sh` for macOS, `start_linux.sh` for Linux, `start_windows.ps1` for Windows) to a convenient location.

2.  **Make the script executable:**
    *   **macOS/Linux:** Open your terminal, navigate to the script's location, and run: `chmod +x start.sh` (or `start_linux.sh`)
    *   **Windows:** No `chmod` needed, but you might need to adjust PowerShell's execution policy. Open PowerShell **as Administrator** and run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` (confirm with 'Y'). You only need to do this once.

3.  **Run the script:**
    *   **macOS:** `./start.sh`
    *   **Linux:** `./start_linux.sh`
    *   **Windows:** Open a **regular** PowerShell window (not as admin), navigate to the script's location, and run `.\start_windows.ps1`. (Alternatively, use `powershell -ExecutionPolicy Bypass -File .\start_windows.ps1` to bypass policy for one run).

4.  **Follow the prompts:** The script will guide you through:
    *   Checking/installing prerequisites (Python, Git, uv, mcp-proxy).
    *   Asking for the Git repository URL to clone.
    *   Asking for your OpenAI API Key (input is hidden).
    *   Setting up the virtual environment and installing dependencies.
    *   Optionally building and installing the tool globally.
    *   Optionally starting the server.

5.  **After the script finishes:**
    *   **Restart your terminal/PowerShell window** to ensure PATH changes are applied.
    *   Review the generated `.env` file in the project directory.
    *   Configure your MCP client (e.g., Cursor) according to the instructions printed by the script and the [Client Configuration](#client-configuration) section below.

## Usage (Manual)

If not using the setup scripts or after manual setup:

### SSE Mode

```bash
# Make sure you are in the project directory with venv activated
uv run server --port 8000
```

### stdio Mode


**Option 1: Build and install globally**

```bash
# 1. Build and install
uv build
uv tool uninstall demcp_browser_mcp 2>/dev/null || true
uv tool install dist/demcp_browser_mcp-*.whl --force

# 2. Run (ensure uv tool path is in PATH)
demcp_browser_mcp run server --port 8000 --stdio --proxy-port 9000
```

## Client Configuration

### SSE Mode Client Configuration

```json
{
  "mcpServers": {
    "demcp_browser_mcp": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### stdio Mode Client Configuration

**If running script directly (Option 1 above):**

```json
{
  "mcpServers": {
    "demcp_browser_mcp_dev": { // Example name
      "command": "python", // Or python3, py.exe etc.
      "args": [
        "server/server.py",
        "--stdio"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key" // Or ensure it's in .env
      },
      "workingDirectory": "/path/to/your/demcp_browser_mcp" // Set this!
    }
  }
}
```

**If running globally installed tool (Option 2 above):**

```json
{
  "mcpServers": {
    "demcp_browser_mcp_tool": { // Example name
      "command": "demcp_browser_mcp", // Or demcp_browser_mcp.exe on Windows
      "args": [
        "run",
        "server",
        "--stdio"
        // --port/--proxy-port usually not needed for direct stdio
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key" // Or ensure it's in .env
      }
    }
  }
}
```

### Config Locations

| Client           | Configuration Path                                                |
| ---------------- | ----------------------------------------------------------------- |
| Cursor           | `./.cursor/mcp.json` (within the project folder) or global settings |
| Windsurf         | `~/.codeium/windsurf/mcp_config.json`                             |
| Claude (Mac)     | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude (Windows) | `%APPDATA%\Claude\claude_desktop_config.json`                     |

## Features

- [x] **Browser Automation**: Control browsers through AI agents
- [x] **Dual Transport**: Support for both SSE and stdio protocols
- [x] **VNC Streaming**: Watch browser automation in real-time
- [x] **Async Tasks**: Execute browser operations asynchronously (Removed in recent updates)

## Local Development

To develop and test the package locally:

1. Ensure prerequisites and dependencies are installed (see [Installation](#installation)).
2. Activate your virtual environment (`source .venv/bin/activate` or similar).
3. Make code changes.
4. Run the server directly for testing:
   ```bash
   python server/server.py --stdio
   ```
5. If installing globally:
   ```bash
   uv build
   uv tool install dist/demcp_browser_mcp-*.whl --force
   demcp_browser_mcp run server --stdio
   ```

## Docker

Using Docker provides a consistent and isolated environment for running the server.

```bash
# Build the Docker image
docker build -t demcp_browser_mcp .

# Run the container with the default VNC password ("browser-use")
# --rm ensures the container is automatically removed when it stops
# -p 8000:8000 maps the server port
# -p 5900:5900 maps the VNC port
# Pass OpenAI API Key as environment variable
docker run --rm -p8000:8000 -p5900:5900 -e OPENAI_API_KEY="your-api-key" demcp_browser_mcp

# Run with a custom VNC password read from a file
echo "your-secure-password" > vnc_password.txt
docker run --rm -p8000:8000 -p5900:5900 \
  -e OPENAI_API_KEY="your-api-key" \
  -v $(pwd)/vnc_password.txt:/run/secrets/vnc_password:ro \
  demcp_browser_mcp
```

*Note: The Docker image runs the server in SSE mode by default. Modify the Dockerfile's CMD instruction for stdio mode.* 

### VNC Viewer

```bash
# Browser-based viewer (run on your host machine)
git clone https://github.com/novnc/noVNC
cd noVNC
./utils/launch.sh --vnc localhost:5900
```

Access `http://localhost:6080/vnc.html` in your browser.
Default password: `browser-use` (unless overridden using the custom password method in Dockerfile)

<div align="center">
  <img width="428" alt="VNC Screenshot" src="https://github.com/user-attachments/assets/45bc5bee-418d-4182-94f5-db84b4fc0b3a" />
  <br><br>
  <img width="428" alt="VNC Screenshot" src="https://github.com/user-attachments/assets/7db53f41-fc00-4e48-8892-f7108096f9c4" />
</div>

## Example

Try asking your AI (configured with the MCP server):

```text
@demcp_browser_mcp run task: open https://news.ycombinator.com and return the top 5 articles as a list
```

## Support

For issues or inquiries: [cobrowser.xyz](https://cobrowser.xyz)

## Star History

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=co-browser/demcp_browser_mcp&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=co-browser/demcp_browser_mcp&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=co-browser/demcp_browser_mcp&type=Date" />
  </picture>
</div>
