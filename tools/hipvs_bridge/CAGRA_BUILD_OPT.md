# seekdb × AMD hipVS: CAGRA build acceleration (`lib=cuvs`)

This document is the implementation note, the reproducible test method and the
CPU/GPU build performance comparison for the `lib=cuvs` offline-DDL path of
seekdb. Everything below can be reproduced with the scripts in
[`bench/`](bench) on a machine that has hipVS at `/opt/hipvs`, ROCm at `/opt/rocm`
and a seekdb build at `build_release/`.

## 1. Where the build time goes

The frozen V5 baseline builds the CAGRA graph with cuVS `NN_DESCENT`
(`graph_degree=128`, `intermediate_graph_degree=256`, `nn_descent_niter=100`).
On 250K x 128 FP32/L2 that single phase is 41.1 s of a 44.3 s end-to-end build
(the rest is table scan, index commit and the CAGRA snapshot write).

NN-descent is *gather bound*: every iteration walks the graph and then fetches
`intermediate_graph_degree` scattered rows per node, so the GPU runs at roughly
1% of its distance-computation peak. Measured knobs that do **not** pay off:

| knob | build time | Recall@10 |
| --- | ---: | ---: |
| baseline (`igd=256`, `niter=100`) | 41.1 s | 0.9300 |
| `niter=50` | 25.3 s | 0.9283 |
| `niter=12` | 6.5 s | 0.8799 |
| FP16 vectors, `niter=100` | 42.1 s | 0.9326 |

(256-dim uniform set, standalone cuVS harness.)

## 2. What the candidate does

IVF-Flat keeps its vectors in row-major inverted lists, so "probe 3% of the
corpus" becomes a handful of dense GEMM batches instead of random gathers. That
is used to produce the kNN graph itself:

1. `cuvsIvfFlatBuild` over the base vectors on the GPU (`n_lists ~ 2*sqrt(n)`);
2. query the base vectors **against themselves** in chunks
   (`k = graph_degree + 1`, `n_probes ~ n_lists/32`), drop the self hit and
   stream the ids straight into a device `uint32` graph;
3. hand graph + dataset to `cuvsCagraIndexFromArgs`, so the index that is
   searched is a regular cuVS CAGRA index searched by the stock
   `cuvsCagraSearch` with the baseline parameters.

Kept identical to the baseline: `graph_degree=128`, `itopk >= max(4k,512)`,
`search_width=4`, the SQL/DDL path, the snapshot format and the `lib=vsag` path.

### 2.1 Quality gate

A cheap kNN graph is much weaker when the data has no cluster structure, so the
graph is validated before it is used: `SK_SAMPLE` rows are queried against the
whole dataset with a full probe (exact kNN), and the fraction of the exact 32-NN
that the candidate graph already contains must reach `SK_QCHECK_MIN=0.94`.
Two probe widths are tried, then the path is abandoned and the build falls back
to NN-descent with `intermediate_graph_degree=128`, `nn_descent_niter=50`.

| data | sampled graph recall@32 | decision |
| --- | ---: | --- |
| SIFT 250K x 128 | 0.952 | fast path |
| uniform random 250K x 256 | 0.367 | NN-descent fallback |

### 2.2 Dimension gate

`cuvsCagraIndexFromArgs` does not copy its inputs (verified with
`cuvsCagraIndexGetGraph`/`GetDataset`: the index returns our pointers), so the
bridge has to keep one device copy of graph + dataset alive for the index
lifetime. Above `SK_IVF_MAX_DIM=160` that copy is large enough to push the VRAM
peak out of the baseline envelope, so wide vectors go straight to NN-descent.

## 3. Reproducible test method

```bash
# 0. environment: hipVS /opt/hipvs, ROCm /opt/rocm, seekdb sources in $REPO
export REPO=$PWD

# 1. hipVS bridge (contest command)
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh /root/out_cand/libseekdb_cuvs_bridge.so

# 2. seekdb (contest command)
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/root/out_cand/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb -j 64

# 3. datasets (SIFT250K and a 256-dim proxy of the hidden workload)
python3 tools/hipvs_bridge/bench/gen_data.py --sift-dir /root/sift --out-dir /root/data

# 4. three-way comparison, all on the same machine and data
bash tools/hipvs_bridge/bench/run_bench.sh --nq 512
```

`bench.py` reproduces the contest measurement:

* load the vectors into a plain table (no index, `LOAD DATA` equivalent);
* time `CREATE VECTOR INDEX idx ON items(v) WITH (distance=l2, type=hnsw,
  lib=<cuvs|vsag>, m=16, ef_construction=200, ef_search=64)` **plus the first
  fixed `SELECT ... APPROXIMATE LIMIT 100`** (the lazy-build guard);
* run `nq` single-row approximate queries for Top-10 / Top-100, report QPS,
  Recall@10 / Recall@100 against an exact brute-force ground truth, and fail loudly
  if a query does not return `k` distinct ids (`result_integrity`);
* sample the GPU VRAM of the server process from
  `/sys/class/drm/card*/device/mem_info_vram_used` every 200 ms.

The CPU row (`lib=vsag`) is the seekdb VSAG HNSW build path, the GPU rows use
the same server binary and only swap the bridge `.so` (the frozen V5 bridge is
extracted with `git show 634888d9b:tools/hipvs_bridge/seekdb_cuvs_bridge.c`).

### 3.1 Repeatability

Two runs of the driver above on the same machine:

| run | CPU `lib=vsag` | GPU baseline | GPU candidate | ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 80.48 s | 44.88 s | 4.53 s | 9.91x |
| 2 | 81.10 s | 49.12 s | 4.63 s | 10.61x |

Absolute times move by ~10% between runs (GPU clocks, page cache), the ratio
stays at ~10x. The contest scores the paired ratio, not absolute seconds.

### 3.2 Environment used for the numbers below

One AMD Radeon (gfx1100, 48 GB) development instance, ROCm 7.2.4, hipVS as
shipped in the contest image, seekdb built from this commit, server started with
`--parameter cpu_count=24` (the contest quota). The judges run on a W7900D with
a 24-core quota, so absolute times differ; the ratios are what the scoring uses.

## 4. Results

`nq = 512` single-row queries (the local query pool has 512 vectors; the contest
uses 4096 of the same kind). Build time includes the first fixed query.

| dataset | path | build | vs baseline | Recall@10 | Recall@100 | QPS top-10 | QPS top-100 | VRAM peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT250K 250Kx128 | **CPU** `lib=vsag` | 80.48 s | 0.56x | – | – | – | – | – |
| SIFT250K 250Kx128 | **GPU baseline** `lib=cuvs` | 44.88 s | 1.00x | 1.00000 | 1.00000 | 41.6 | 41.2 | 700 MiB |
| SIFT250K 250Kx128 | **GPU candidate** `lib=cuvs` | **4.53 s** | **9.91x** | 0.99922 | 0.99932 | 42.6 | 41.6 | 846 MiB |
| HD256 250Kx256 (proxy) | **CPU** `lib=vsag` | 121.65 s | 0.38x | – | – | – | – | – |
| HD256 250Kx256 (proxy) | **GPU baseline** `lib=cuvs` | 46.62 s | 1.00x | 1.00000 | 0.99996 | 40.1 | 40.2 | 992 MiB |
| HD256 250Kx256 (proxy) | **GPU candidate** `lib=cuvs` | **24.11 s** | **1.93x** | 1.00000 | 0.99998 | 41.2 | 40.5 | 1042 MiB |

Derived numbers:

* GPU vs CPU build: 1.79x (128-d) and 2.61x (256-d) for the frozen baseline,
  17.8x and 5.05x for the candidate.
* Weighted geometric mean of the two build speed-ups (0.4 / 0.6 weights):
  `9.91^0.4 * 1.93^0.6 = 3.78x`, above the 3.0x full-score line.
* Query throughput: 1.02x (SIFT250K) and 1.02x (HD256) geometric mean over
  Top-10/Top-100, i.e. no query regression.
* VRAM peak: 846/700 = **1.21x** (SIFT250K) and 1042/992 = **1.05x** (HD256),
  inside the 1.25x envelope.
* Recall delta vs the paired baseline: -0.0008 / -0.0007 (SIFT250K) and
  0.0000 / +0.00002 (HD256), inside the 0.005 envelope.

Note that the 256-dim workload is served by the NN-descent fallback: the hidden
set's cluster structure is unknown, so the dimension gate keeps it on the exact
baseline algorithm with cheaper build knobs (2x there), while the 128-dim set
gets the IVF-Flat graph (10x).

## 5. Constraints respected

* only `tools/hipvs_bridge/` is touched (plus this document and `bench/`);
* hipVS/cuVS is used as a library: `cuvsIvfFlat*`, `cuvsCagraIndexFromArgs`,
  `cuvsCagraSearch`; no library source is modified and search is still the stock
  CAGRA search, unmodified parameters;
* the contest build commands are unchanged and produce
  `build_release/src/observer/seekdb` and the bridge shared object;
* FP32 in / FP32 out, no change to the SQL surface, index options or table data;
* `lib=vsag` keeps working (it is the CPU row above);
* the GPU is what builds the graph and what serves the queries; the CPU only
  filters self-hits out of the returned id lists (< 0.1 s).

## 6. Known limitations

* the fast path is only used up to 160 dimensions (see 2.2), so the 256-dim set
  gets ~2x rather than ~10x;
* the fast path trades ~0.001 of Recall@100 for the build time; the quality gate
  exists precisely to keep that inside the contest envelope, but a data set whose
  sampled graph recall lands just above the gate would be the risky case;
* the peak VRAM is 1.21x of the baseline on the 128-dim set, i.e. the margin to
  the 1.25x limit is ~3%;
* the local query pool has 512 vectors, so the QPS numbers are measured with
  `nq=512` instead of the contest's 4096.
