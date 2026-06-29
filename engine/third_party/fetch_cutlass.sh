#!/usr/bin/env bash
# Fetch the CUTLASS/CuTe C++ headers needed by src/attn_cute.cu into
# engine/third_party/cutlass/include (gitignored, ~26MB). The headers ship
# inside the nvidia-cutlass PyPI wheel, which downloads far more reliably than a
# git clone or codeload tarball behind a slow/mirrored proxy.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/cutlass/include"
[ -f "$DEST/cute/tensor.hpp" ] && { echo "CUTLASS already present at $DEST"; exit 0; }

TMP="$(mktemp -d)"
echo "Downloading nvidia-cutlass wheel ..."
pip download nvidia-cutlass --no-deps -d "$TMP" ${PIP_PROXY:+--proxy "$PIP_PROXY"}
unzip -q -o "$TMP"/nvidia_cutlass-*.whl -d "$TMP/wheel"
mkdir -p "$HERE/cutlass"
cp -r "$TMP/wheel/cutlass_library/source/include" "$DEST"
rm -rf "$TMP"
echo "CUTLASS headers installed at $DEST"
