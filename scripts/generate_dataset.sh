#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Dino Core V1 - Dataset Generation ==="
echo "Project root: $PROJECT_ROOT"
echo ""

uv run python scripts/semantic_patching/generate_dataset.py "$@"
