#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}
GPU_ARCH=${HIPVS_GPU_ARCH:-gfx1100}
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}

[[ -x "$HIPCC" ]] || { echo "hipcc is not executable: $HIPCC" >&2; exit 2; }
[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2
  exit 2
}
[[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || {
  echo "libcuvs_c.so is missing under $HIPVS_ROOT/lib" >&2
  exit 2
}

mkdir -p "$(dirname "$OUTPUT")"
OBJECTS=$(mktemp -d)
trap 'rm -f -- "$OBJECTS/bridge.o" "$OBJECTS/pack_graph.o"; rmdir -- "$OBJECTS"' EXIT

"$HIPCC" \
  -x c -c \
  -fPIC \
  -pthread \
  -O2 \
  -D__HIP_PLATFORM_AMD__ \
  -DSK_DEVICE_GRAPH_PACK \
  -I"$HIPVS_ROOT/include" \
  -I"$ROCM_ROOT/include" \
  "$SCRIPT_DIR/seekdb_cuvs_bridge.c" \
  -o "$OBJECTS/bridge.o"

"$HIPCC" -x hip -c --offload-arch="$GPU_ARCH" -fPIC -O2 \
  -I"$ROCM_ROOT/include" "$SCRIPT_DIR/pack_graph.hip" -o "$OBJECTS/pack_graph.o"

"$HIPCC" -shared -pthread --offload-arch="$GPU_ARCH" \
  "$OBJECTS/bridge.o" "$OBJECTS/pack_graph.o" \
  -L"$HIPVS_ROOT/lib" \
  -L"$ROCM_ROOT/lib" \
  -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -lcuvs_c \
  -o "$OUTPUT"

echo "built $OUTPUT"
