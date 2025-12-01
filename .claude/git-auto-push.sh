#!/usr/bin/env bash
# =============================================================================
# git-auto-push.sh - 自动 cleanup + commit + push（只提交核心文件）
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "=============================================="
echo "[git-auto] Auto commit/push pipeline"
echo "[git-auto] Time: $TIMESTAMP"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 1: 运行清理脚本
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/.claude/cleanup.sh" ]; then
    echo "[git-auto] Running cleanup..."
    bash "$PROJECT_ROOT/.claude/cleanup.sh"
fi

# -----------------------------------------------------------------------------
# Step 2: 只添加核心文件（不是 git add -A）
# -----------------------------------------------------------------------------
echo "[git-auto] Adding core files only..."

# 核心代码目录
git add aad_edge_pushing/aad/ 2>/dev/null || true
git add aad_edge_pushing/pde/ 2>/dev/null || true
git add aad_edge_pushing/__init__.py 2>/dev/null || true

# edge_pushing 核心文件（排除测试和生成文件）
git add aad_edge_pushing/edge_pushing/__init__.py 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/algo*.py 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/algo*.pyx 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/algo*.cpp 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/algo*.hpp 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/symm_sparse*.py 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/symm_sparse*.pyx 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/symm_sparse*.cpp 2>/dev/null || true
git add aad_edge_pushing/edge_pushing/setup*.py 2>/dev/null || true

# 配置文件
git add .gitignore 2>/dev/null || true
git add .claude/cleanup.sh 2>/dev/null || true
git add .claude/git-auto-push.sh 2>/dev/null || true
git add .claude/settings.json 2>/dev/null || true
git add .claude/commands/*.md 2>/dev/null || true

# 移除不应该跟踪的文件（如果之前误添加）
git reset HEAD -- aad_edge_pushing/edge_pushing/test_*.py 2>/dev/null || true
git reset HEAD -- aad_edge_pushing/edge_pushing/*.html 2>/dev/null || true
git reset HEAD -- aad_edge_pushing/edge_pushing/*.c 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 3: 检查是否有变更
# -----------------------------------------------------------------------------
if git diff --cached --quiet; then
    echo "[git-auto] No changes to commit. Exiting."
    exit 0
fi

echo "[git-auto] Changes to be committed:"
git diff --cached --stat

# -----------------------------------------------------------------------------
# Step 4: Commit
# -----------------------------------------------------------------------------
CHANGED_FILES=$(git diff --cached --name-only | head -5)
FILE_COUNT=$(git diff --cached --name-only | wc -l)

COMMIT_MSG="auto: Claude Code session changes [$TIMESTAMP]

Changed files ($FILE_COUNT total):
$(echo "$CHANGED_FILES" | sed 's/^/  - /')
$([ "$FILE_COUNT" -gt 5 ] && echo "  ... and $((FILE_COUNT - 5)) more files")
"

git commit -m "$COMMIT_MSG"

# -----------------------------------------------------------------------------
# Step 5: Push
# -----------------------------------------------------------------------------
echo "[git-auto] Pushing to remote..."
CURRENT_BRANCH=$(git branch --show-current)

if ! git push origin "$CURRENT_BRANCH" 2>/dev/null; then
    echo "[git-auto] Push failed, trying pull --rebase..."
    git pull --rebase origin "$CURRENT_BRANCH" || exit 1
    git push origin "$CURRENT_BRANCH"
fi

echo "=============================================="
echo "[git-auto] Done! Pushed to $CURRENT_BRANCH"
echo "=============================================="
