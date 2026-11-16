#!/bin/bash
# Reproducible driver for the numbers in ../CAGRA_BUILD_OPT.md.
#
#   bash tools/hipvs_bridge/bench/run_bench.sh [--nq 512] [--only sift250k|hd256]
#
# Measures, on the same machine, the same data and the same server binary:
#   1. lib=vsag  : the CPU HNSW build path of seekdb            ("CPU" row)
#   2. lib=cuvs  : GPU CAGRA with the frozen V5 baseline bridge ("GPU baseline")
#   3. lib=cuvs  : GPU CAGRA with this repository's bridge      ("GPU candidate")
#
# The bridge is swapped through LD_LIBRARY_PATH, so no rebuild of seekdb is
# needed between the two GPU rows.
#
# Requirements: seekdb built at build_release/ (contest cmake command),
# /opt/hipvs, /opt/rocm, mysql client, numpy, data from gen_data.py at $DATA_DIR.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
DATA_DIR=${DATA_DIR:-/root/data}
OUT=${OUT:-/root/bench_results.jsonl}
BASE_COMMIT=${BASE_COMMIT:-634888d9b037153fa143d2c913ba4b25503e1d62}
BASE_DIR=${BASE_DIR:-/root/vdata}
NQ=512
ONLY=all
while [[ $# -gt 0 ]]; do
  case $1 in
    --nq) NQ=$2; shift 2;;
    --only) ONLY=$2; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done
PY=${PY:-python3}
BRIDGE_SRC="$REPO/tools/hipvs_bridge/seekdb_cuvs_bridge.c"

build_bridge() {  # $1 = output dir, $2 = "base" | "cand"
  mkdir -p "$1"
  if [[ "$2" == "base" ]]; then
    git -C "$REPO" show "$BASE_COMMIT:tools/hipvs_bridge/seekdb_cuvs_bridge.c" > "$BRIDGE_SRC"
    HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
      bash "$REPO/tools/hipvs_bridge/build.sh" "$1/libseekdb_cuvs_bridge.so" >/dev/null 2>&1
    git -C "$REPO" checkout -- tools/hipvs_bridge/seekdb_cuvs_bridge.c
  else
    HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
      bash "$REPO/tools/hipvs_bridge/build.sh" "$1/libseekdb_cuvs_bridge.so" >/dev/null 2>&1
  fi
}

start_seekdb() {  # $1 = bridge dir, $2 = tag
  pkill -9 -f "base-dir=${BASE_DIR}" 2>/dev/null
  sleep 3
  ( cd "$REPO" && env LD_LIBRARY_PATH="$1:/opt/hipvs/lib:/opt/rocm/lib" \
      setsid nohup build_release/src/observer/seekdb --nodaemon \
      --base-dir="$BASE_DIR" --data-dir="$BASE_DIR/store" \
      --redo-dir="$BASE_DIR/store/redo" --port=2881 --parameter cpu_count=24 \
      > "/root/obs_$2.log" 2>&1 < /dev/null & )
  for _ in $(seq 1 120); do
    sleep 1
    mysql -h127.0.0.1 -P2881 -uroot -N -s -e 'SELECT 1' >/dev/null 2>&1 && return 0
  done
  echo "seekdb did not become ready"; return 1
}

run_set() {  # $1 = dataset prefix, $2 = dim
  local p=$1 d=$2
  [[ "$ONLY" != "all" && "$ONLY" != "$p" ]] && return 0
  echo "=== $p ($d-d) ==="
  start_seekdb /root/out_base base || return 1
  # CPU row: loads the table itself (no index) and times the VSAG HNSW build
  "$PY" "$(dirname "$0")/bench.py" --x "$DATA_DIR/${p}_x.f32" --q "$DATA_DIR/${p}_q.f32" \
    --dim "$d" --nq "$NQ" --lib vsag --tag "$p/cpu-vsag" --skip-query --out "$OUT"
  "$PY" "$(dirname "$0")/bench.py" --x "$DATA_DIR/${p}_x.f32" --q "$DATA_DIR/${p}_q.f32" \
    --dim "$d" --nq "$NQ" --lib cuvs --tag "$p/gpu-baseline" --skip-load --out "$OUT"
  start_seekdb /root/out_cand cand || return 1
  "$PY" "$(dirname "$0")/bench.py" --x "$DATA_DIR/${p}_x.f32" --q "$DATA_DIR/${p}_q.f32" \
    --dim "$d" --nq "$NQ" --lib cuvs --tag "$p/gpu-candidate" --skip-load --out "$OUT"
}

mkdir -p /root/out_base /root/out_cand "$BASE_DIR"
: > "$OUT"
build_bridge /root/out_base base
build_bridge /root/out_cand cand
run_set sift250k 128
run_set hd256 256
echo "--- results ---"
cat "$OUT"
