#!/bin/bash
set -e

# Release script for workflow-engine
# Usage: ./release.sh <version>
# Example: ./release.sh 0.3.3

if [ -z "$1" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.3.3"
    exit 1
fi

VERSION=$1

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in semver format (e.g., 0.3.3)"
    exit 1
fi

# Ensure we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "Error: Must be on main branch (currently on $BRANCH)"
    exit 1
fi

# Ensure working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean"
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Update version in pyproject.toml
echo "Updating version to $VERSION..."
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Run tests
echo "Running tests..."
uv run pytest -q

# Commit version bump
echo "Committing version bump..."
git add pyproject.toml
git commit -m "Bump version to $VERSION"

# Create and push tag
echo "Creating tag v$VERSION..."
git tag "v$VERSION"

# Push commit and tag
echo "Pushing to origin..."
git push
git push origin "v$VERSION"

# Create GitHub release
echo "Creating GitHub release..."
gh release create "v$VERSION" \
    --title "workflow-engine v$VERSION" \
    --generate-notes

echo "Released v$VERSION successfully!"
