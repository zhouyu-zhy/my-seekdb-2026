# Performance branch validation

Base: 5d6f53493f0c9b04c6de1c4edf42d33fbe46a235.
This branch changes only source and tests. Both Prompt files, Dockerfile,
entrypoint and public protocol cases retain their base contents.

Build the base in a separate directory and keep its AVX2 executable as
BASELINE. Build the candidate using the unchanged minivectordb/Makefile.

    make -C minivectordb -j2
    taskset -c 0,1 python3 minivectordb/tests/bench_paired.py --baseline "$BASELINE" --candidate minivectordb/minivectordb-avx2 --rounds 3 --output /tmp/paired-benchmark.json
    taskset -c 0,1 python3 minivectordb/tests/check_engine.py --baseline "$BASELINE" --candidate minivectordb/minivectordb-avx2
    python3 minivectordb/tests/check_simd_boundaries.py minivectordb/minivectordb minivectordb/minivectordb-avx2
    python3 tests/bench_one.py --binary minivectordb/minivectordb-avx2

Run comparisons without concurrent builds or benchmarks. The default workload
uses 10,000 rows, 128 dimensions, batches of 100 and 1,000 TopK-10 queries;
half the queries filter by a scalar equality. The seed, dim, rows
and queries CLI options support other workloads.

## What changed

1. A commit no longer issues `fdatasync` per transaction block.  The contest
   crash model is process termination, so a complete block that reached the
   kernel's page cache is durable; the block is still fully written, flushed
   and self-checking before the response is sent.  Checkpoint and catalog
   writes keep their device flush, and shutdown still flushes the WAL.
2. The queue benchmark's TopK shape
   (`SELECT cols, L2_DISTANCE(v, q) AS d ... [WHERE col = literal] ORDER BY d LIMIT k`)
   now runs on a dedicated path: a 7-bit columnar code array scores every row
   with an AVX2 byte dot product, the best `k + margin` rows are rescored with
   the float32 kernel, and the final rows are rescored with the reference
   double kernel.  A Cauchy-Schwarz bound on the quantization error proves
   that no excluded row can enter the answer; if the bound fails, the query
   falls back to the general executor, so results stay exact.
3. `l2DistancePtr` moved to `src/distance.h` so both paths share the same
   accumulation order and therefore the same reported bits.

## Measured on the development machine (median of three paired rounds)

Workload: 10,000 rows x 128 dims, batch 100, 1,000 TopK-10 queries (half
filtered), taskset 0-1, WSL2 ext4.

| metric | base 5d6f5349 | this branch | change |
| --- | --- | --- | --- |
| write stage | 0.494 s | 0.054 s | 9.1x faster |
| query stage | 0.152 s | 0.128 s | 1.19x faster |
| peak RSS | 14.5 MiB | 19.4 MiB | +4.9 MiB |
| response equality | - | all 3,000 responses byte-identical | |

The query stage includes the driver's stdin/stdout round trip, which is about
60 us per request in this VM and dominates the remaining difference; the
engine itself runs the TopK query in 53 us against 76 us for the base.

## Verification performed

- `tests/check_engine.py` passes: 3,000 byte-identical responses against the
  base binary, independent brute-force neighbours, ties, +/-1e38 and
  denormal vectors, illegal vectors, failed transaction statements, SIGKILL,
  torn WAL tails, interrupted recovery under a 64 MiB address-space cap, and
  identical WAL bytes plus bidirectional old/new recovery.
- `tests/check_simd_boundaries.py` passes for 16 dimension sizes.
- The official `protocol_check/run.py` passes all 20 public cases (native
  command mode; the Docker image still needs the owner's final run).
- A standalone harness (`tests/` does not ship it) compared 50 random queries
  over 10,000 x 128 rows against a brute-force double scan: 0 mismatches and
  0 fallbacks.

The merged commit still requires the owner's final Docker/public-case run
before the single submission.
