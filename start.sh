#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions ---
print_info() {
  echo "INFO: $1"
}

print_warning() {
  echo "WARNING: $1"
}

print_error() {
  echo "ERROR: $1" >&2
  exit 1
}

# Function to check if a command exists and install if not
# Usage: check_install <command_name> <install_brew_package_name> <install_uv_package_name>
check_or_install() {
  local cmd="$1"
  local brew_pkg="$2"
  local uv_pkg="$3"
  local install_cmd=""

  if ! command -v "$cmd" &> /dev/null; then
    print_warning "'$cmd' not found."
    if [[ -n "$brew_pkg" ]] && command -v brew &> /dev/null; then
      print_info "Attempting to install '$brew_pkg' using Homebrew..."
      if brew install "$brew_pkg"; then
        print_info "'$brew_pkg' installed successfully via Homebrew."
      else
        print_error "Failed to install '$brew_pkg' using Homebrew. Please install it manually."
      fi
    elif [[ -n "$uv_pkg" ]] && command -v uv &> /dev/null; then
       print_info "Attempting to install '$uv_pkg' using uv..."
       if uv tool install "$uv_pkg"; then
         print_info "'$uv_pkg' installed successfully via uv."
       else
         print_error "Failed to install '$uv_pkg' using uv. Please install it manually."
       fi
    else
        # Specific handling for uv itself
        if [ "$cmd" == "uv" ]; then
            print_info "Attempting to install uv..."
            if curl -LsSf https://astral.sh/uv/install.sh | sh; then
                print_info "'uv' installed successfully."
                print_warning "You might need to add '~/.cargo/bin' (or similar, check uv output) to your PATH."
                print_warning "Please restart your terminal or run 'source ~/.zshrc' (or ~/.bash_profile) after this script finishes."
                # Attempt to add uv to the current script's PATH for subsequent steps
                export PATH="$HOME/.cargo/bin:$PATH"
                if ! command -v uv &> /dev/null; then
                   print_error "Failed to find 'uv' even after installation attempt. Please check PATH and retry."
                fi
            else
                print_error "Failed to install 'uv'. Please install it manually from https://github.com/astral-sh/uv"
            fi
        else
             print_error "Cannot install '$cmd'. Please install it manually. Requires 'brew' for '$brew_pkg' or 'uv' for '$uv_pkg'."
        fi
    fi
  else
    print_info "'$cmd' is already installed."
  fi
}


# --- Main Setup ---

print_info "Starting demcp_browser_mcp setup for macOS..."

# 1. Check System Prerequisites
print_info "Checking system prerequisites..."

## Homebrew
if ! command -v brew &> /dev/null; then
    print_warning "'brew' (Homebrew) not found."
    print_info "Homebrew is recommended for installing Python and other tools."
    read -p "Do you want to attempt to install Homebrew now? (Requires sudo) [y/N]: " install_brew
    if [[ "$install_brew" =~ ^[Yy]$ ]]; then
        print_info "Running the Homebrew installation script..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Attempt to add brew to PATH for this script session (may vary based on install location)
        eval "$(/opt/homebrew/bin/brew shellenv)"
         if ! command -v brew &> /dev/null; then
             print_error "Homebrew installation finished, but 'brew' command not found. Please check installation and PATH."
         fi
    else
        print_warning "Skipping Homebrew installation. Some dependencies might fail if not installed manually."
    fi
else
    print_info "'brew' is already installed."
fi

## Python 3.11+
print_info "Checking for Python 3.11+..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_info "Found Python version: $PY_VERSION"
    if [[ "$(python3 -c 'import sys; print(sys.version_info >= (3, 11))')" != "True" ]]; then
        print_warning "Python version is older than 3.11."
        check_or_install "python@3.11" "python@3.11" "" # Check_or_install will handle brew install
    else
         print_info "Python version is sufficient."
    fi
else
    print_warning "Python 3 not found."
    check_or_install "python@3.11" "python@3.11" "" # Check_or_install will handle brew install
fi
# Ensure python3 command points to a valid version now if installed
if ! command -v python3 &> /dev/null; then
    print_error "Failed to find or install a suitable Python 3 version. Please install Python 3.11+ manually."
fi


## Xcode Command Line Tools
print_info "Checking for Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    print_warning "Xcode Command Line Tools not found or not configured."
    print_info "Attempting to install. Please follow the prompts in the new window."
    xcode-select --install || print_warning "Xcode Command Line Tools installation might have failed or was cancelled. Some packages may fail to build."
    # Wait a bit for the user to potentially finish the install prompt
    print_info "Waiting for potential Xcode tools installation... (Press Enter to continue if you cancelled or it finished quickly)"
    read -p "" dummy_read
else
    print_info "Xcode Command Line Tools are installed."
fi

## uv
check_or_install "uv" "" ""

## mcp-proxy
check_or_install "mcp-proxy" "" "mcp-proxy"


# 2. Project Setup
print_info "Setting up the project..."

repo_url="https://github.com/demcp/brower-use-mcp"
print_info "Using hardcoded repository URL: $repo_url"

# Extract project directory name from URL
project_dir=$(basename "$repo_url" .git)

## Clone Repo
if [ -d "$project_dir" ]; then
    print_warning "Directory '$project_dir' already exists. Skipping git clone."
else
    print_info "Cloning repository $repo_url..."
    if ! git clone "$repo_url"; then
        print_error "Failed to clone repository. Please check the URL and your network connection."
    fi
fi

## Enter Project Directory
cd "$project_dir" || print_error "Failed to enter project directory '$project_dir'."
print_info "Changed directory to $(pwd)"

## Create and Activate Virtual Environment
print_info "Setting up Python virtual environment using uv..."
uv venv || print_error "Failed to create virtual environment."
print_info "Activating virtual environment for subsequent steps in this script..."
# Note: This activates for the script's subshell only.
source .venv/bin/activate || print_error "Failed to activate virtual environment."

# 3. Install Dependencies
print_info "Installing project dependencies..."

## Python Dependencies
print_info "Running 'uv sync' to install Python dependencies..."
uv sync || print_error "Failed to install Python dependencies with 'uv sync'."

## Playwright
print_info "Installing Playwright library..."
uv pip install playwright || print_warning "Failed to install Playwright library (might already be installed)."
print_info "Installing Playwright browser dependencies (Chromium)..."
uv run playwright install --with-deps --no-shell chromium || print_error "Failed to install Playwright browsers."

# 4. Configuration
print_info "Configuring environment..."

## Create .env file
env_file=".env"
print_info "Creating $env_file file..."

# Prompt securely for OpenAI API Key
unset openai_api_key # Ensure variable is clean
while [[ -z "$openai_api_key" ]]; do
    read -sp "Please enter your OpenAI API Key: " openai_api_key
    echo # Add a newline after the prompt
    if [[ -z "$openai_api_key" ]]; then
        print_warning "OpenAI API Key cannot be empty."
    fi
done

# Write API key to .env file (overwrite if exists)
echo "OPENAI_API_KEY=$openai_api_key" > "$env_file"

# Add comments for optional variables
echo "" >> "$env_file"
echo "# Optional: Uncomment and set if needed" >> "$env_file"
echo "# CHROME_PATH=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" >> "$env_file"
echo "# OPENAI_MODEL=gpt-4o" >> "$env_file"
echo "# OPENAI_API_BASE=your_custom_openai_api_base_url" >> "$env_file"
# Add other optional vars from your config if desired

print_info "$env_file created successfully with API Key."
print_warning "Review $env_file to set optional variables like CHROME_PATH if needed."

# 5. (Optional) Build and Install Globally
install_globally="n"
read -p "Do you want to build and install 'demcp_browser_mcp' as a global tool using uv? (Useful for running outside the project dir) [y/N]: " install_globally

# Define the command to run, adding dummy ports for the workaround
if [[ "$install_globally" =~ ^[Yy]$ ]]; then
    print_info "Building the project wheel..."
    uv build || print_error "Failed to build the project."
    print_info "Installing the tool globally using uv..."
    # Find the built wheel file
    wheel_file=$(find dist -name "demcp_browser_mcp-*.whl" | head -n 1)
     if [[ -z "$wheel_file" ]]; then
         print_error "Could not find the built wheel file in dist directory."
     fi
    uv tool install "$wheel_file" --force || print_error "Failed to install the tool globally."
    print_info "'demcp_browser_mcp' installed globally."
    print_warning "Remember to ensure the uv tool bin path ('$HOME/.cargo/bin' or similar) is in your main shell's PATH."
    run_command="demcp_browser_mcp run server --stdio --port 8001 --proxy-port 9001" # Workaround: Add dummy ports
else
    print_info "Skipping global installation. You'll need to run the server using 'python server/server.py'."
    run_command="python server/server.py --stdio --port 8001 --proxy-port 9001" # Workaround: Add dummy ports
fi


# --- Completion ---
echo ""
print_info "------------------------------------------"
print_info "Setup Complete!"
print_info "------------------------------------------"
echo ""
print_info "Next Steps:"
echo "1. If this is the first time running, **restart your terminal** or run 'source ~/.zshrc' (or your shell's equivalent) to ensure 'uv' and potentially 'demcp_browser_mcp' (if installed globally) are in your PATH."
echo "2. Review the '.env' file in the '$project_dir' directory to ensure settings (like CHROME_PATH if needed) are correct."
echo "3. Configure Cursor:"
echo "   - Go to Settings -> MCP Servers -> Edit in settings.json"
if [[ "$install_globally" =~ ^[Yy]$ ]]; then
    echo "   - Add/Update a server configuration using command: 'demcp_browser_mcp' and args: ['run', 'server', '--stdio']"
else
    echo "   - Add/Update a server configuration using command: 'python' and args: ['server/server.py', '--stdio'] (ensure workingDirectory is set to the project root)."
fi
echo "   - Make sure to add the OPENAI_API_KEY to the 'env' section in the Cursor settings as well, or ensure the server picks it up from the .env file."
echo ""

# 6. (Optional) Start Server
start_now="n"
read -p "Do you want to attempt to start the server now in stdio mode? (Requires terminal restart first; Configure Cursor separately) [y/N]: " start_now

if [[ "$start_now" =~ ^[Yy]$ ]]; then
    print_warning "Make sure you have restarted your terminal before running this (if needed for PATH changes)."

    # Check for netcat (nc) command, needed for port checking
    if ! command -v nc &> /dev/null; then
        print_warning "'nc' (netcat) command not found, needed for port checking."
        if command -v brew &> /dev/null; then
            print_info "Attempting to install 'netcat' using Homebrew..."
            if brew install netcat; then
                print_info "'netcat' installed successfully."
            else
                print_error "Failed to install 'netcat'. Cannot check for available ports automatically."
            fi
        else
            print_error "'brew' not found. Cannot install 'netcat'. Cannot check for available ports automatically."
        fi
        # Re-check after install attempt
        if ! command -v nc &> /dev/null; then
           print_error "'nc' command still not found after installation attempt."
        fi
    fi

    # Find available port starting from 8000
    port=8000
    while nc -z 127.0.0.1 $port &>/dev/null; do
        print_info "Port $port is in use, trying next..."
        port=$((port + 1))
        if [ $port -gt 9000 ]; then # Safety break to avoid infinite loop
            print_error "Could not find an available port between 8000 and 9000."
            exit 1
        fi
    done
    found_port=$port
    proxy_port=$((found_port + 1)) # Assign proxy port relative to found port
    print_info "Found available port: $found_port (using $proxy_port for proxy if needed by server internal logic)"

    # Construct the final command with the found ports
    # Check if the global install was chosen earlier to select the base command
    final_run_command=""
    if [[ "$install_globally" =~ ^[Yy]$ ]]; then
        final_run_command="demcp_browser_mcp run server --stdio --port $found_port --proxy-port $proxy_port"
    else
        final_run_command="python server/server.py --stdio --port $found_port --proxy-port $proxy_port"
    fi

    print_info "Attempting to start the server using command: '$final_run_command'"
    print_warning "The script will now run the server. Press Ctrl+C to stop it when finished."
    # Execute the command
    eval "$final_run_command"
else
    print_info "Server not started. After restarting terminal, you can start it manually using (example with port 8000):"
    # Determine the base command for the manual instructions
    manual_base_command=""
     if [[ "$install_globally" =~ ^[Yy]$ ]]; then
        manual_base_command="demcp_browser_mcp run server"
    else
        manual_base_command="python server/server.py"
    fi
    print_info "  cd $(pwd)" # Use pwd as project_dir might not be set if clone was skipped
    print_info "  source .venv/bin/activate"
    print_info "  $manual_base_command --stdio --port 8000 --proxy-port 8001 # Adjust port if needed"
fi

exit 0