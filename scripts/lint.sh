#!/usr/bin/env bash
# Format + lint the C++/CUDA engine sources. Usage:
#   scripts/lint.sh format    # rewrite files in place (clang-format)
#   scripts/lint.sh check     # CI gate: fail if any file is mis-formatted
#   scripts/lint.sh tidy      # clang-tidy over host .cpp (needs build/compile_commands.json)
#   scripts/lint.sh           # = check + tidy
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_DIR=engine/src
BUILD_DIR=engine/build
# clang-format handles every source kind, including CUDA .cu/.cuh (layout only).
ALL_SRC=$(find "$SRC_DIR" -maxdepth 1 -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) | sort)
# clang-tidy parses real translation units; restrict it to host .cpp (device .cu
# would need a clang CUDA toolchain to consume nvcc's compile_commands entries).
HOST_TU=$(find "$SRC_DIR" -maxdepth 1 -type f -name '*.cpp' | sort)

fmt()   { echo "$ALL_SRC" | xargs clang-format -i; echo "formatted $(echo "$ALL_SRC" | wc -l) files"; }
check() { echo "$ALL_SRC" | xargs clang-format --dry-run --Werror && echo "format OK"; }
tidy()  {
  [ -f "$BUILD_DIR/compile_commands.json" ] || {
    echo "no $BUILD_DIR/compile_commands.json — configure cmake first" >&2; exit 1; }
  echo "$HOST_TU" | xargs run-clang-tidy -p "$BUILD_DIR" -quiet
}

case "${1:-all}" in
  format) fmt ;;
  check)  check ;;
  tidy)   tidy ;;
  all)    check; tidy ;;
  *) echo "usage: $0 {format|check|tidy}" >&2; exit 2 ;;
esac
