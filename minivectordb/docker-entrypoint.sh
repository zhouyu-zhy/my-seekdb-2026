#!/bin/sh
set -e
# Pick the widest build the host CPU actually supports.  Every build speaks the
# same protocol; only the distance kernel differs.
flags=""
vendor=""
if [ -r /proc/cpuinfo ]; then
    flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
    vendor=$(grep -m1 '^vendor_id' /proc/cpuinfo 2>/dev/null | awk '{print $3}' || true)
fi
has() { echo "$flags" | tr ' ' '\n' | grep -qx "$1"; }
# 512-bit byte dot products only pay off where the core has full-width 512-bit
# units (Intel server parts).  AMD double-pumps them, so the 256-bit build is
# faster there and is used instead.
if [ "$vendor" = "GenuineIntel" ] && [ -x /app/minivectordb-avx512 ] \
   && has avx512f && has avx512bw && has avx512vl && has avx512vnni; then
    exec /app/minivectordb-avx512 "$@"
fi
if has avx2 && has fma; then
    exec /app/minivectordb-avx2 "$@"
fi
exec /app/minivectordb "$@"
