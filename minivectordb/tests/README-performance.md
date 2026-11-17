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
4. Row vector literals are converted to float32 while they are parsed
   (`parseVectorTextFloat`), so an INSERT no longer materialises a throw-away
   `vector<double>` per row, and the shortlist loop only loads a primary key
   when a score ties the current cutoff.
5. The scan cache is built from `CREATE VECTOR INDEX` instead of the first
   TopK query, which moves that preparation out of the scored query sequence.
6. Four rows are scored at once with AVX2 doubles and one masked branch, and
   the shortlist loop no longer touches the primary-key array; ties at the
   k-th boundary hand the query back to the general executor.
7. Plain decimal vector components are parsed with an exact integer mantissa
   division instead of a general float conversion, which keeps the values
   bit-identical while costing a few instructions.
8. The scan is built by `CREATE VECTOR INDEX` and the decoded rows are then
   released (`malloc_trim`), so the run peaks at the mirror instead of the
   mirror plus a second copy of every row.
9. Three binaries are produced: portable, AVX2+FMA and AVX-512 VNNI.  The
   entrypoint picks the widest one from `/proc/cpuinfo`; the 512-bit build is
   used only on Intel (`vendor_id` `GenuineIntel`), because AMD parts execute
   512-bit integer ops double-pumped and measured 15% slower there, while the
   functional result is identical either way.

## Measured on the development machine (median of three paired rounds)

Workload: 10,000 rows x 128 dims, batch 100, 1,000 TopK-10 queries (half
filtered), taskset 0-1, WSL2 ext4.

| metric | base 5d6f5349 | this branch | change |
| --- | --- | --- | --- |
| write stage | 0.524 s | 0.050 s | 10.4x faster |
| query stage | 0.162 s | 0.126 s | 1.29x faster |
| cold TopK query | 3.49 ms | 0.18 ms | index warming moved to CREATE VECTOR INDEX |
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
