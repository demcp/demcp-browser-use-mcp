# 2. Project Setup
Print-Info "Setting up the project..."

## Define Git Repo URL (Hardcoded)
$repo_url = "https://github.com/LyiZri/demcp_browser_use"
Print-Info "Using hardcoded repository URL: $repo_url"

# Extract project directory name from URL
$project_dir = [System.IO.Path]::GetFileNameWithoutExtension($repo_url)

## Clone Repo
if (Test-Path -Path $project_dir) {
    Print-Warning "Directory '$project_dir' already exists. Skipping git clone."
} else {
    Print-Info "Cloning repository $repo_url..."
    try {
        git clone $repo_url
    } catch {
         Print-Error "Failed to clone repository. Please check the URL, your network connection, and Git setup. Error: $($_.Exception.Message)"
    }
}

## Enter Project Directory
try {
    Set-Location -Path $project_dir
    Print-Info "Changed directory to $(Get-Location)"
} catch {
    Print-Error "Failed to enter project directory '$project_dir'."
}

# 3. Install Dependencies
# ... rest of the script ...
