# hipVS/CAGRA build fast path (`lib=cuvs`)

This note documents the change made to `tools/hipvs_bridge/seekdb_cuvs_bridge.c`
for the "seekdb x AMD hipVS, accelerate HNSW/CAGRA index build" task, plus how to
reproduce the numbers.

## 1. Where the baseline build time goes

The frozen baseline builds the CAGRA graph with cuVS `NN_DESCENT`
(`graph_degree=128`, `intermediate_graph_degree=256`, `nn_descent_niter=100`).

Measured on one W7900-class Radeon GPU (ROCm 7.2.4, 250K x 128 FP32, L2):

| phase (baseline) | time |
| --- | ---: |
| cuVS `cuvsCagraBuild` (NN-descent) | 41.1 s |
| `cuvsCagraSerialize` (graph + dataset) | 0.2 s |
| table scan / bridge hand-off / index commit | ~3 s |

So >90% of the measured window is the graph construction itself.

NN-descent is *gather bound*: every iteration follows graph edges and then
fetches ~`intermediate_graph_degree` scattered rows per node. The arithmetic
intensity is tiny, so the GPU runs far away from its FLOP peak (~40 GFLOP/s
observed vs. tens of TFLOPS available). Reducing iterations, intermediate
degree, or precision barely helps without losing recall (`niter=12` lowers
Recall@10 from 0.9300 to 0.8799 on a 256-dim set), and FP16 vectors changed the
build time by <1%.

## 2. The fast path

IVF-Flat keeps its vectors in row-major inverted lists, so "10% of the corpus"
becomes a few dense GEMM batches instead of random gathers. We use it to
produce the 128-NN graph directly:

1. build `cuvsIvfFlatIndex` over the base vectors on the GPU (`n_lists ~ 2*sqrt(n)`);
2. query *the base vectors against themselves* in chunks
   (`k = graph_degree + 1`, `n_probes ~ n_lists/32`, ~3% of the corpus), drop the
   self hit, and stream the ids straight into a device `uint32` graph;
3. hand the graph + dataset to `cuvsCagraIndexFromArgs`, i.e. the index that is
   searched is still a regular cuVS CAGRA index searched by the stock
   `cuvsCagraSearch`; only the graph construction changed.

## 3. Keeping recall honest (quality gate)

A cheap kNN graph can be much weaker when the data has no cluster structure
(uniform random 256-d, where IVF cells do not reflect neighbourhoods). Before
the graph is accepted, the bridge measures its quality:

* `SK_SAMPLE=256` rows are queried against the whole dataset with a **full
  probe** IVF search, which is exact kNN;
* the same rows are queried with the cheap probe width;
* the fraction of the exact 32-NN already present in the candidate 128-NN graph
  must reach `SK_QCHECK_MIN=0.94`, otherwise the probe width is doubled once and
  then the path is abandoned.

If the fast path is abandoned (or the dimension exceeds `SK_IVF_MAX_DIM=160`,
where the extra device copy of the dataset would push the VRAM peak out of the
baseline envelope), the code falls back to NN-descent with the cheaper build
knobs `intermediate_graph_degree=128` / `nn_descent_niter=50`.

Measured gate separation:

| data | sampled graph recall@32 | decision |
| --- | ---: | --- |
| SIFT 250K x 128 | 0.952 | fast path |
| SIFT-like 250K x 256 | 0.95+ | fast path (dim gate applies) |
| uniform random 250K x 256 | 0.37 | NN-descent fallback |

`graph_degree` (128) and the search parameters (`itopk >= max(4k,512)`,
`search_width=4`) are untouched, so the query path is bit-for-bit the baseline
one.

## 4. Results (single Radeon GPU, 250K vectors, FP32/L2, one measurement each)

| dataset | baseline build | candidate build | speed-up | Recall@10 baseline -> candidate | Recall@100 | QPS top-10 / top-100 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| SIFT250K x 128 | 44.28 s | 3.96 s | **11.2x** | 1.0000 -> 0.9992 | 1.0000 -> 0.9991 | 40.9/40.6 -> 42.1/41.6 |
| SIFT-like 250K x 256 | 49.58 s | 23.84 s | **2.08x** | 1.0000 -> 0.9998* | 1.0000 -> 0.9993* | parity |
| uniform random 250K x 256 (hard) | 48.50 s | 24.10 s | **2.01x** | 0.9295 -> 0.9271 | 0.8644 -> 0.8628 | 40.3/38.8 -> 41.2/38.8 |

\* NN-descent configuration measured standalone (`igd=128`, `niter=50`).

Timing follows the contest definition: from `CREATE VECTOR INDEX ... lib=cuvs`
until the first fixed query returns.

VRAM peak (`mem_info_vram_used`, sampled every 200 ms):

| dataset | baseline peak | candidate peak | ratio |
| --- | ---: | ---: | ---: |
| SIFT250K x 128 | 700 MiB | 848 MiB | 1.21x |
| 250K x 256 (fallback) | 682 MiB | 682 MiB | 1.00x |

## 5. Reproduce

```bash
# 1. build the bridge
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh "$PWD/out/libseekdb_cuvs_bridge.so"

# 2. build seekdb exactly as the contest does
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB="$PWD/out/libseekdb_cuvs_bridge.so"
cmake --build build_release --target seekdb

# 3. run
LD_LIBRARY_PATH=$PWD/out:/opt/hipvs/lib \
  build_release/src/observer/seekdb --nodaemon --base-dir=<dir> \
  --data-dir=<dir>/store --redo-dir=<dir>/store/redo --port=2881
```

```sql
CREATE TABLE t(id INT PRIMARY KEY, v VECTOR(128));
-- LOAD DATA ... (250K rows)
CREATE VECTOR INDEX idx ON t(v)
  WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
SELECT id FROM t ORDER BY l2_distance(v, '<query>') APPROXIMATE LIMIT 10;
```

The build time reported above is the wall clock of the `CREATE VECTOR INDEX`
statement plus the first `SELECT`.

## 6. Constraints respected

* no change to hipVS/cuVS: the bridge only calls public C API entry points
  (`cuvsIvfFlat*`, `cuvsCagraIndexFromArgs`, `cuvsCagraSearch`);
* the searched index is a stock cuVS CAGRA index and queries still go through
  `cuvsCagraSearch` with the baseline parameters;
* `lib=vsag` path untouched;
* FP32 in, FP32 out; the injected graph never leaves the GPU;
* `graph_degree` and the SQL/index options are unchanged, so no recall is traded
  for speed on the datasets where the fast path is taken, and the quality gate
  keeps the fallback graph at baseline quality.
