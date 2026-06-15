#!/bin/bash
# Generate the axon.patch file from the vllm fork.
#
# Usage:
#   bash install/vllm/generate_patch.sh /path/to/vllm-fork
#
# This creates patches relative to the pinned vllm base commit:
#   - axon.patch contains Python-only hunks for wheel installs.
#   - axon-csrc.patch contains optional C++ hunks for source builds.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/axon.patch"

# Must match the VLLM_COMMIT in install_vllm.sh
VLLM_BASE_COMMIT="2a69949bdadf0e8942b7a1619b229cb475beef20"
VLLM_BASE_SHORT="2a69949bd"

VLLM_FORK_DIR="${1:-}"
if [ -z "$VLLM_FORK_DIR" ]; then
    echo "Usage: bash install/vllm/generate_patch.sh /path/to/vllm-fork"
    exit 1
fi

if [ ! -d "$VLLM_FORK_DIR/.git" ]; then
    echo "ERROR: $VLLM_FORK_DIR is not a git repository."
    exit 1
fi

cd "$VLLM_FORK_DIR"

# Verify the base commit exists
if ! git cat-file -e "$VLLM_BASE_COMMIT" 2>/dev/null; then
    echo "ERROR: Base commit $VLLM_BASE_SHORT not found in $VLLM_FORK_DIR."
    echo "       Run: git fetch origin"
    exit 1
fi

# Generate two patches:
# 1. Python-only patch (applied to wheel installs via install_vllm.sh)
# 2. C++ patch (applied only when building from source — wheels ship precompiled .so)
echo "==> Generating patch: ${VLLM_BASE_SHORT}..HEAD"
git diff "$VLLM_BASE_COMMIT" -- \
    'vllm/**/*.py' \
    ':!tests/' \
    ':!docs/' \
    ':!benchmarks/' \
    > "$PATCH_FILE"

CSRC_PATCH_FILE="$SCRIPT_DIR/axon-csrc.patch"
git diff "$VLLM_BASE_COMMIT" -- \
    'csrc/*.cpp' \
    'csrc/*.h' \
    > "$CSRC_PATCH_FILE"
CSRC_NUM_FILES=$(grep -c '^diff --git' "$CSRC_PATCH_FILE" 2>/dev/null || echo 0)
if [ "$CSRC_NUM_FILES" -eq 0 ]; then
    rm -f "$CSRC_PATCH_FILE"
else
    echo "    C++ patch: $CSRC_PATCH_FILE ($CSRC_NUM_FILES files)"
    echo "    NOTE: csrc patch only applies to source builds. Recompile with:"
    echo "      cd /path/to/vllm && pip install -e ."
fi

NUM_FILES=$(grep -c '^diff --git' "$PATCH_FILE" 2>/dev/null || echo 0)
PATCH_SIZE=$(wc -c < "$PATCH_FILE" | tr -d ' ')

echo "==> Patch generated: $PATCH_FILE"
echo "    Base commit: $VLLM_BASE_SHORT"
echo "    Files changed: $NUM_FILES"
echo "    Patch size: $PATCH_SIZE bytes"

if [ "$NUM_FILES" -eq 0 ]; then
    echo "    (No Python changes relative to base — patch is empty)"
fi
