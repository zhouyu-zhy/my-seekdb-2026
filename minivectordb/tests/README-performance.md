# Performance branch validation

Base: 5d6f53493f0c9b04c6de1c4edf42d33fbe46a235.
This branch changes only source and tests. Both Prompt files, Dockerfile,
entrypoint and public protocol cases retain their base contents.

Build the base in a separate directory and keep its AVX2 executable as
BASELINE. Build the candidate using the unchanged minivectordb/Makefile.

    make -C minivectordb -j2
    taskset -c 0,1 python3 minivectordb/tests/bench_paired.py --baseline "$BASELINE" --candidate minivectordb/minivectordb-avx2 --rounds 5 --output /tmp/paired-benchmark.json
    taskset -c 0,1 python3 minivectordb/tests/check_engine.py --baseline "$BASELINE" --candidate minivectordb/minivectordb-avx2
    python3 minivectordb/tests/check_simd_boundaries.py minivectordb/minivectordb minivectordb/minivectordb-avx2

Run comparisons without concurrent builds or benchmarks. The default workload
uses 10,000 rows, 128 dimensions, batches of 100 and 1,000 TopK-10 queries;
half the queries filter by a scalar equality. The seed, dim, rows
and queries CLI options support other workloads. Cold-index cost is recorded
separately. The native paired results are evidence, not an official score prediction.

The regression test checks full response equality against the base, independent
brute-force neighbors, transactional cache invalidation, SIGKILL, torn WAL tails,
interrupted recovery with a WAL over 64 MiB under a 64 MiB address-space cap,
and identical WAL bytes plus recovery in both old-to-new and new-to-old directions.

The SIMD test covers 200 independent brute-force queries across both binaries,
16 dimension sizes (including 31/32/33, 63/64/65 and 4096/4097), subnormal and
large FP32 magnitudes, ties and filters. Each binary also performs 582 SQL/JSON
string round trips and rejects 1,950 raw JSON control-byte cases at span boundaries.
Scalar and SIMD neighbor IDs must match exactly. Their distance outputs use
a 1e-12 relative tolerance because FMA reduction may change final rounding bits.

The results/2026-11-17-performance directory contains the measured five-pair
benchmark, regression output, boundary output, source hashes and native public
results. Both native executables passed all 20 public cases. The merged commit
still requires the owner's final Docker/public-case run before the single submission.
