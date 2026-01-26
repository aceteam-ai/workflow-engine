#!/bin/bash
set -e

# Manual deploy script for aceteam-workflow-engine
# This script builds and uploads the package to PyPI manually.
# Use this for emergency deploys or when GitHub Actions is unavailable.
#
# Prerequisites:
#   - PyPI API token configured in ~/.pypirc or via TWINE_PASSWORD env var
#   - uv installed
#
# Usage: ./deploy.sh
#
# For TestPyPI (dry run): PYPI_REPOSITORY=testpypi ./deploy.sh

REPOSITORY=${PYPI_REPOSITORY:-pypi}

echo "Building package..."
rm -rf dist/
uv build

echo ""
echo "Built artifacts:"
ls -la dist/

echo ""
echo "Running tests before deploy..."
uv run pytest -q

echo ""
echo "Running linting..."
uv run ruff check .

echo ""
echo "Running type checking..."
uv run pyright

echo ""
read -p "Deploy to $REPOSITORY? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Uploading to $REPOSITORY..."
if [ "$REPOSITORY" = "testpypi" ]; then
    uv run twine upload --repository testpypi dist/*
else
    uv run twine upload dist/*
fi

echo ""
echo "Deployed successfully!"
