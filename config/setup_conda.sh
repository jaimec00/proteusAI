#!/bin/bash

# Exit on errors in pipelines or commands
set -euo pipefail

# Function: print error messages to stderr
_error() {
    printf "Error: %s\n" "$1" >&2
}

# Function: check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check if conda is already available
if command_exists conda; then
    # If conda is in PATH, show its version and exit
    echo "Found existing conda installation:"
    conda --version || _error "Failed to run 'conda --version', but 'conda' is in PATH."
    exit 0
fi

# Step 2: Prompt the user
read -r -p "Conda not found. Would you like to download and install Miniconda? [y/N]: " response
response=${response,,}  # lowercase
if [[ ! "$response" =~ ^(y|yes)$ ]]; then
    echo "Aborting installation."
    exit 0
fi

# Step 3: Detect OS
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"

# Normalize architecture strings
case "$ARCH" in
    x86_64|amd64) ARCH_TAG="x86_64" ;;
    arm64|aarch64) 
        if [[ "$OS_NAME" == "Darwin" ]]; then
            ARCH_TAG="arm64"
        else
            # On Linux ARM64, Miniconda may not have official builds; warn user
            ARCH_TAG="aarch64"
        fi
        ;;
    *)
        _error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Step 4: Determine installer filename based on OS
case "$OS_NAME" in
    Linux)
        INSTALLER_NAME="Miniconda3-latest-Linux-${ARCH_TAG}.sh"
        ;;
    Darwin)
        # macOS installer naming uses 'MacOSX'
        INSTALLER_NAME="Miniconda3-latest-MacOSX-${ARCH_TAG}.sh"
        ;;
    *)
        _error "Unsupported OS: $OS_NAME"
        exit 1
        ;;
esac

BASE_URL="https://repo.anaconda.com/miniconda"
INSTALLER_URL="${BASE_URL}/${INSTALLER_NAME}"

# Step 5: Download the installer
TMP_DIR="$(mktemp -d)"
INSTALLER_PATH="${TMP_DIR}/${INSTALLER_NAME}"

echo "Downloading installer from ${INSTALLER_URL}..."
if command_exists curl; then
    curl -fsSL "${INSTALLER_URL}" -o "${INSTALLER_PATH}" || {
        _error "Download failed via curl."
        exit 1
    }
elif command_exists wget; then
    wget -q "${INSTALLER_URL}" -O "${INSTALLER_PATH}" || {
        _error "Download failed via wget."
        exit 1
    }
else
    _error "Neither 'curl' nor 'wget' is available for downloading."
    exit 1
fi

# Make installer executable
chmod +x "${INSTALLER_PATH}"

# Step 6: Prompt for install prefix or use default
DEFAULT_PREFIX="$HOME/miniconda3"
read -r -p "Install prefix [default: ${DEFAULT_PREFIX}]: " PREFIX_INPUT
INSTALL_PREFIX="${PREFIX_INPUT:-$DEFAULT_PREFIX}"

echo "Installing Miniconda into ${INSTALL_PREFIX}..."
bash "${INSTALLER_PATH}" -b -p "${INSTALL_PREFIX}" || {
    _error "Installer execution failed."
    exit 1
}

# Step 7: Initialize conda for current shell and future shells
# Source the conda hook for this session
CONDA_SH="${INSTALL_PREFIX}/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
elif [[ -f "${INSTALL_PREFIX}/bin/conda" ]]; then
    # Fallback: use 'conda shell.bash hook'
    eval "$("${INSTALL_PREFIX}/bin/conda" shell.bash hook)"
else
    _error "Cannot find conda initialization script."
    echo "You may need to manually add '${INSTALL_PREFIX}/bin' to your PATH."
    exit 1
fi

# Run conda init to modify ~/.bashrc (or shell config)
echo "Initializing conda for Bash (this modifies your ~/.bashrc)..."
conda init bash || _error "conda init failed; you may need to run it manually."

# Step 8: Cleanup
echo "Cleaning up installer..."
rm -rf "${TMP_DIR}"

echo "Miniconda installation complete."
echo "Please restart your shell or run 'source ~/.bashrc' to start using conda."
exit 0