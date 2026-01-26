#!/bin/bash
set -e

# Release script for aceteam-workflow-engine
# Usage: ./release.sh [--major | --minor | --patch | --rc] [VERSION]
#
# Examples:
#   ./release.sh           # Auto-bump patch (1.0.0 -> 1.0.1)
#   ./release.sh --minor   # Auto-bump minor (1.0.0 -> 1.1.0)
#   ./release.sh --major   # Auto-bump major (1.0.0 -> 2.0.0)
#   ./release.sh --rc      # Add/bump release candidate (1.0.0 -> 1.0.1rc1, 1.0.1rc1 -> 1.0.1rc2)
#   ./release.sh 2.0.0     # Explicit version

BUMP_TYPE="patch"
EXPLICIT_VERSION=""
IS_RC=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --major)
            BUMP_TYPE="major"
            shift
            ;;
        --minor)
            BUMP_TYPE="minor"
            shift
            ;;
        --patch)
            BUMP_TYPE="patch"
            shift
            ;;
        --rc)
            IS_RC=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./release.sh [--major | --minor | --patch | --rc] [VERSION]"
            echo ""
            echo "Options:"
            echo "  --major    Bump major version (1.0.0 -> 2.0.0)"
            echo "  --minor    Bump minor version (1.0.0 -> 1.1.0)"
            echo "  --patch    Bump patch version (1.0.0 -> 1.0.1) [default]"
            echo "  --rc       Create release candidate (1.0.0 -> 1.0.1rc1)"
            echo "  VERSION    Explicit version to release"
            echo ""
            echo "Examples:"
            echo "  ./release.sh              # 1.0.0 -> 1.0.1"
            echo "  ./release.sh --minor      # 1.0.0 -> 1.1.0"
            echo "  ./release.sh --rc         # 1.0.0 -> 1.0.1rc1"
            echo "  ./release.sh --rc         # 1.0.1rc1 -> 1.0.1rc2"
            echo "  ./release.sh 2.0.0        # Explicit version"
            exit 0
            ;;
        *)
            EXPLICIT_VERSION="$1"
            shift
            ;;
    esac
done

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

# Get latest tag
LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
LATEST_VERSION="${LATEST_TAG#v}"
echo "Latest version: $LATEST_VERSION"

# Parse version components
parse_version() {
    local version="$1"
    # Handle rc versions: 1.0.0rc1 -> major=1, minor=0, patch=0, rc=1
    if [[ "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(rc([0-9]+))?$ ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        MINOR="${BASH_REMATCH[2]}"
        PATCH="${BASH_REMATCH[3]}"
        RC="${BASH_REMATCH[5]:-0}"
    else
        echo "Error: Cannot parse version '$version'"
        exit 1
    fi
}

# Calculate new version
if [ -n "$EXPLICIT_VERSION" ]; then
    NEW_VERSION="$EXPLICIT_VERSION"
else
    parse_version "$LATEST_VERSION"

    if [ "$IS_RC" = true ]; then
        if [ "$RC" -gt 0 ]; then
            # Already an RC, just bump RC number
            RC=$((RC + 1))
        else
            # Not an RC, bump patch and start RC at 1
            PATCH=$((PATCH + 1))
            RC=1
        fi
        NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}rc${RC}"
    else
        # Clear any RC suffix for regular releases
        RC=0
        case $BUMP_TYPE in
            major)
                MAJOR=$((MAJOR + 1))
                MINOR=0
                PATCH=0
                ;;
            minor)
                MINOR=$((MINOR + 1))
                PATCH=0
                ;;
            patch)
                PATCH=$((PATCH + 1))
                ;;
        esac
        NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
    fi
fi

# Validate new version format
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]]; then
    echo "Error: Invalid version format '$NEW_VERSION'"
    exit 1
fi

echo "New version: $NEW_VERSION"

# Get commit history since last tag
echo ""
echo "Changes since $LATEST_TAG:"
echo "----------------------------------------"
COMMIT_LOG=$(git log "$LATEST_TAG"..HEAD --pretty=format:"- %s (%h)" --no-merges 2>/dev/null || echo "- Initial release")
echo "$COMMIT_LOG"
echo "----------------------------------------"
echo ""

# Confirm with user
read -p "Proceed with release v$NEW_VERSION? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Update version in pyproject.toml and __init__.py
echo "Updating version to $NEW_VERSION..."
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" src/workflow_engine/__init__.py

# Run tests
echo "Running tests..."
uv run pytest -q

# Run linting
echo "Running linting..."
uv run ruff check .

# Commit version bump
echo "Committing version bump..."
git add pyproject.toml src/workflow_engine/__init__.py
git commit -m "Bump version to $NEW_VERSION"

# Create and push tag
echo "Creating tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Push commit and tag
echo "Pushing to origin..."
git push
git push origin "v$NEW_VERSION"

# Build release notes with commit history
RELEASE_NOTES=$(cat <<EOF
## What's Changed

$COMMIT_LOG

**Full Changelog**: https://github.com/aceteam-ai/workflow-engine/compare/$LATEST_TAG...v$NEW_VERSION
EOF
)

# Create GitHub release
echo "Creating GitHub release..."
if [[ "$NEW_VERSION" == *"rc"* ]]; then
    gh release create "v$NEW_VERSION" \
        --title "aceteam-workflow-engine v$NEW_VERSION" \
        --notes "$RELEASE_NOTES" \
        --prerelease
else
    gh release create "v$NEW_VERSION" \
        --title "aceteam-workflow-engine v$NEW_VERSION" \
        --notes "$RELEASE_NOTES"
fi

echo ""
echo "✅ Released v$NEW_VERSION successfully!"
echo "   https://github.com/aceteam-ai/workflow-engine/releases/tag/v$NEW_VERSION"
