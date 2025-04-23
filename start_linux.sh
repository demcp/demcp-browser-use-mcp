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

# Function to detect package manager
detect_pkg_manager() {
  if command -v apt &> /dev/null; then
    echo "apt"
  elif command -v dnf &> /dev/null; then
    echo "dnf"
  elif command -v yum &> /dev/null; then
    echo "yum" # Older Fedora/CentOS
  elif command -v pacman &> /dev/null; then
     echo "pacman" # Arch
  elif command -v zypper &> /dev/null; then
     echo "zypper" # openSUSE
  else
    echo "unknown"
  fi
}

# Function to check if a command exists and install if not
# Usage: check_install <command_name> <apt_package_name> <dnf_package_name> <pacman_package_name> <zypper_package_name> <uv_package_name>
check_or_install() {
  local cmd="$1"
  local apt_pkg="$2"
  local dnf_pkg="$3"
  local pacman_pkg="$4"
  local zypper_pkg="$5"
  local uv_pkg="$6"
  local pkg_manager=$(detect_pkg_manager)
  local install_cmd=""

  if ! command -v "$cmd" &> /dev/null; then
    print_warning "'$cmd' not found."
    case "$pkg_manager" in
      apt)
        if [[ -n "$apt_pkg" ]]; then
          print_info "Attempting to install '$apt_pkg' using apt..."
          sudo apt update || print_warning "apt update failed, proceeding anyway..."
          if sudo apt install -y "$apt_pkg"; then
             print_info "'$apt_pkg' installed successfully via apt."
          else
             print_error "Failed to install '$apt_pkg' using apt. Please install it manually."
          fi
        fi
        ;;
      dnf|yum)
         if [[ -n "$dnf_pkg" ]]; then
           print_info "Attempting to install '$dnf_pkg' using $pkg_manager..."
           if sudo "$pkg_manager" install -y "$dnf_pkg"; then
              print_info "'$dnf_pkg' installed successfully via $pkg_manager."
           else
              print_error "Failed to install '$dnf_pkg' using $pkg_manager. Please install it manually."
           fi
         fi
         ;;
      pacman)
         if [[ -n "$pacman_pkg" ]]; then
            print_info "Attempting to install '$pacman_pkg' using pacman..."
            if sudo pacman -Syu --noconfirm "$pacman_pkg"; then
               print_info "'$pacman_pkg' installed successfully via pacman."
            else
               print_error "Failed to install '$pacman_pkg' using pacman. Please install it manually."
            fi
         fi
         ;;
      zypper)
         if [[ -n "$zypper_pkg" ]]; then
            print_info "Attempting to install '$zypper_pkg' using zypper..."
            if sudo zypper install -y "$zypper_pkg"; then
               print_info "'$zypper_pkg' installed successfully via zypper."
            else
               print_error "Failed to install '$zypper_pkg' using zypper. Please install it manually."
            fi
         fi
         ;;
      *)
        # Try installing via UV if specified
        if [[ -n "$uv_pkg" ]] && command -v uv &> /dev/null; then
           print_info "Attempting to install '$uv_pkg' using uv..."
           if uv tool install "$uv_pkg"; then
             print_info "'$uv_pkg' installed successfully via uv."
             return # Successfully installed via uv
           else
             print_error "Failed to install '$uv_pkg' using uv. Please install it manually."
           fi
        fi

        # Specific handling for uv itself
        if [ "$cmd" == "uv" ]; then
            print_info "Attempting to install uv..."
            if curl -LsSf https://astral.sh/uv/install.sh | sh; then
                print_info "'uv' installed successfully."
                print_warning "You might need to add '~/.cargo/bin' (or similar, check uv output) to your PATH."
                print_warning "Please restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc etc.) after this script finishes."
                # Attempt to add uv to the current script's PATH for subsequent steps
                export PATH="$HOME/.cargo/bin:$PATH"
                if ! command -v uv &> /dev/null; then
                   print_error "Failed to find 'uv' even after installation attempt. Please check PATH and retry."
                fi
                return # Successfully installed uv
            else
                print_error "Failed to install 'uv'. Please install it manually from https://github.com/astral-sh/uv"
            fi
        fi

        # If we get here, package manager unknown/unsupported for this package
        print_error "Unsupported package manager '$pkg_manager' or package not specified for it. Cannot install '$cmd'. Please install it manually."
        ;;
    esac

    # Verify installation even if package manager succeeded, command might differ from package
    if ! command -v "$cmd" &> /dev/null; then
       print_error "Installation command for '$cmd' finished, but command still not found. Please check PATH or install manually."
    fi
  else
    print_info "'$cmd' is already installed."
  fi
}

# --- Main Setup ---
print_info "Starting demcp_browser_mcp setup for Linux..."

# 0. Detect Package Manager
PKG_MANAGER=$(detect_pkg_manager)
print_info "Detected package manager: $PKG_MANAGER"
if [ "$PKG_MANAGER" == "unknown" ]; then
    print_warning "Could not detect a known package manager (apt, dnf, yum, pacman, zypper). You may need to install prerequisites manually."
fi

# 1. Check System Prerequisites
print_info "Checking system prerequisites..."

## Basic Tools
check_or_install "curl" "curl" "curl" "curl" "curl" ""
check_or_install "git" "git" "git" "git" "git" ""

## Build Tools (Needed for some Python packages)
print_info "Checking for build tools..."
case "$PKG_MANAGER" in
    apt) check_or_install "gcc" "build-essential" "" "" "" "" ;; # gcc is usually part of build-essential
    dnf|yum) check_or_install "gcc" "" "gcc" "" "" "" ; check_or_install "make" "" "make" "" "" "";; # Simplistic check, Development Tools group better
    pacman) check_or_install "gcc" "" "" "base-devel" "" "";; # base-devel includes gcc, make etc.
    zypper) check_or_install "gcc" "" "" "" "patterns-devel-base-devel_basis" "";;
    *) print_warning "Cannot automatically check/install build tools for $PKG_MANAGER. Ensure you have gcc, make, etc." ;;
esac


## Python 3.11+ (and pip/venv)
print_info "Checking for Python 3.11+..."
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python" # Try plain python
    if ! command -v $PYTHON_CMD &> /dev/null; then
         print_warning "Neither 'python3' nor 'python' found."
         # Install python3 (adjust packages per distro)
         check_or_install "python3" "python3 python3-pip python3-venv" "python3 python3-pip python3-wheel" "python python-pip" "python3 python3-pip python3-devel" ""
         PYTHON_CMD="python3" # Assume python3 after install
         if ! command -v $PYTHON_CMD &> /dev/null; then print_error "Failed to install Python 3."; fi
    fi
fi

PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_info "Found Python version: $PY_VERSION (using '$PYTHON_CMD')"
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    print_warning "Python version ($PY_VERSION) is older than 3.11. Attempting to install/ensure Python 3.11+..."
    # This is tricky as distros might not have 3.11+ readily available.
    # Best effort: ensure python3 is installed, user might need deadsnakes PPA (Ubuntu) or build from source.
    check_or_install "python3" "python3 python3-pip python3-venv" "python3 python3-pip python3-wheel" "python python-pip" "python3 python3-pip python3-devel" ""
    # Recheck version after ensuring python3 package
    PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
     if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
          print_error "Installed Python version ($PY_VERSION) is still less than 3.11. Please install Python 3.11+ manually (e.g., using deadsnakes PPA on Ubuntu, or building from source) and ensure '$PYTHON_CMD' points to it."
     fi
else
     print_info "Python version is sufficient."
     # Ensure pip and venv are available for this version
     check_or_install "pip3" "python3-pip" "python3-pip" "python-pip" "python3-pip" ""
     check_or_install "$PYTHON_CMD -m venv --help" "python3-venv" "python3-devel" "python" "python3-devel" "" > /dev/null # Check venv module works

fi


## uv
check_or_install "uv" "" "" "" "" "" # Installs via curl

## mcp-proxy
check_or_install "mcp-proxy" "" "" "" "" "mcp-proxy" # Installs via uv


# 2. Project Setup
print_info "Setting up the project..."

## Define Git Repo URL (Hardcoded)
repo_url="https://github.com/LyiZri/demcp_browser_use"
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
print_warning "Playwright will now attempt to install system dependencies. This might fail."
print_warning "If browser launch fails later, you may need to install dependencies manually."
print_warning "See: https://playwright.dev/docs/intro#linux"
if uv run playwright install --with-deps --no-shell chromium; then
    print_info "Playwright browser dependency installation attempted."
else
    print_error "Playwright browser dependency installation failed. Please check the output and install missing libraries manually (e.g., using 'sudo apt install <library-name>' or 'sudo dnf install <library-name>'). See Playwright Linux docs."
fi

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
echo "# Typical Linux Chrome path (may vary):" >> "$env_file"
echo "# CHROME_PATH=/usr/bin/google-chrome-stable" >> "$env_file"
echo "# OPENAI_MODEL=gpt-4o" >> "$env_file"
echo "# OPENAI_API_BASE=your_custom_openai_api_base_url" >> "$env_file"
# Add other optional vars from your config if desired

print_info "$env_file created successfully with API Key."
print_warning "Review $env_file to set optional variables like CHROME_PATH if needed."

# 5. (Optional) Build and Install Globally
install_globally="n"
read -p "Do you want to build and install 'demcp_browser_mcp' as a global tool using uv? (Useful for running outside the project dir) [y/N]: " install_globally

run_command="python server/server.py --stdio" # Default to running script directly

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
    run_command="demcp_browser_mcp run server --stdio"
else
    print_info "Skipping global installation. You'll need to run the server using 'python server/server.py'."
fi


# --- Completion ---
echo ""
print_info "------------------------------------------"
print_info "Setup Complete!"
print_info "------------------------------------------"
echo ""
print_info "Next Steps:"
print_warning "1. IMPORTANT: **Restart your terminal** or run 'source ~/.bashrc' (or your shell's equivalent like ~/.zshrc) NOW to ensure PATH changes (for uv, etc.) take effect."
print_info "2. Review the '.env' file in the '$(pwd)' directory to ensure settings (like CHROME_PATH if needed) are correct."
print_info "3. If you encounter browser launch issues, you may need to manually install missing system libraries listed by Playwright (see https://playwright.dev/docs/intro#linux)."
print_info "4. Configure Cursor:"
echo "   - Go to Settings -> MCP Servers -> Edit in settings.json"
if [[ "$install_globally" =~ ^[Yy]$ ]]; then
    echo "   - Add/Update a server configuration using command: 'demcp_browser_mcp' and args: ['run', 'server', '--stdio']"
else
    echo "   - Add/Update a server configuration using command: 'python' (or 'python3') and args: ['server/server.py', '--stdio'] (ensure workingDirectory is set to the project root)."
fi
echo "   - Make sure to add the OPENAI_API_KEY to the 'env' section in the Cursor settings as well, or ensure the server picks it up from the .env file."
echo ""

# 6. (Optional) Start Server
start_now="n"
read -p "Do you want to attempt to start the server now in stdio mode? (Requires terminal restart first; Configure Cursor separately) [y/N]: " start_now

if [[ "$start_now" =~ ^[Yy]$ ]]; then
    print_warning "Make sure you have restarted your terminal before running this."
    print_info "Attempting to start the server using command: '$run_command'"
    print_warning "The script will now run the server. Press Ctrl+C to stop it when finished."
    # Execute the command
    eval "$run_command"
else
    print_info "Server not started. After restarting terminal, you can start it manually using:"
    print_info "  cd $(pwd)"
    print_info "  source .venv/bin/activate"
    print_info "  $run_command"
fi

exit 0