#!/usr/bin/env bash
# =============================================================================
# cleanup.sh - 清理临时文件、cache、build artifacts
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[cleanup] Cleaning $PROJECT_ROOT..."

# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Build artifacts
find . -type f -name "*.so" -delete 2>/dev/null || true
find . -type f -name "*.o" -delete 2>/dev/null || true

# Temp files
find . -type f -name "*.log" -delete 2>/dev/null || true
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Jupyter
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# pytest/mypy cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

echo "[cleanup] Done!"
