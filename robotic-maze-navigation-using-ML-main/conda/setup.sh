#!/usr/bin/env bash
# Setup script for Linux and macOS
# To add dependencies, edit environment.yml, then re-run this script.

set -e

ENV_NAME="maze-project"

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda env update --name "${ENV_NAME}" --file conda/environment.yml --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create --file conda/environment.yml
fi

echo ""
echo "Done! Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
