#!/bin/sh
set -e
# Pick the AVX2 build when the host CPU supports it, otherwise fall back to the
# portable build.  Both binaries implement exactly the same protocol.
if [ -r /proc/cpuinfo ] && grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null; then
    exec /app/minivectordb-avx2 "$@"
fi
exec /app/minivectordb "$@"
