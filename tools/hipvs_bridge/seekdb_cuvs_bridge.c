/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Degree of the graph that is searched. Kept equal to the contest baseline so
 * the query path (and its cost) is unchanged. */
#define SK_GRAPH_DEGREE 128
/* NN-descent fallback knobs: 128 intermediate degree x 50 iterations keeps the
 * graph quality at baseline level while roughly halving the build time. */
#define SK_NN_ITER 50
#define SK_IVF_MIN_N 4096
/* The fast path keeps one extra device copy of the dataset alive (cuVS builds
 * the index over the buffers we hand it). Above this dimension that copy is
 * large enough to push the VRAM peak past the baseline envelope, so wide
 * vectors stay on the NN-descent path. */
#define SK_IVF_MAX_DIM 160
/* Sampled quality gate for the fast path (see sk_build_ivf_graph). */
#define SK_QCHECK_MIN 0.94
#define SK_SAMPLE 256
#define SK_EXACT_K 32
#define SK_CHUNK 1024
/* Above this dimension the device copy of the dataset that the CAGRA index is
 * built over is kept in FP16 (half of the FP32 footprint). The contest allows
 * internal FP16 as long as the external FP32 semantics and the recall gates
 * hold; queries are converted to FP16 for the search as well. */
#define SK_F16_DIM 100000  /* disabled: IVF-Flat rejects FP16 here */

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  int  f16;                     /* dataset of this index is FP16 */
  /* device buffers the index was constructed from (kept alive until the index
   * is destroyed, because cuVS may reference them without copying) */
  void *graph_buf;
  void *data_buf;
} sk_cuvs_index;

/* float -> IEEE half, round to nearest even */
static unsigned short sk_f2h(float f)
{
  unsigned int x;
  unsigned int sign, man, h;
  int exp;
  memcpy(&x, &f, sizeof(x));
  sign = (x >> 16) & 0x8000u;
  exp = (int)((x >> 23) & 0xffu) - 127 + 15;
  man = x & 0x7fffffu;
  if (exp <= 0) {
    unsigned int shift;
    if (exp < -10) return (unsigned short)sign;
    man |= 0x800000u;
    shift = (unsigned int)(14 - exp);
    h = man >> shift;
    if ((man >> (shift - 1)) & 1u) ++h;
    return (unsigned short)(sign | h);
  }
  if (exp >= 31) return (unsigned short)(sign | 0x7c00u);
  h = sign | ((unsigned int)exp << 10) | (man >> 13);
  if (man & 0x1000u) ++h;
  return (unsigned short)h;
}

static void sk_mk_tensor(DLManagedTensor *t, long *shape, void *data,
                         long rows, long cols, int code, int bits, int device)
{
  shape[0] = rows; shape[1] = cols;
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = device ? kDLCUDA : kDLCPU;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = code;
  t->dl_tensor.dtype.bits = bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
}

/* --------------------------------------------------------------------------
 * NN-descent build: the contest baseline algorithm with cheaper build knobs.
 * ------------------------------------------------------------------------ */
static int sk_build_nn_descent(sk_cuvs_index *h, const float *base, long n, long dim)
{
  DLManagedTensor dset; long ds[2] = {n, dim};
  sk_mk_tensor(&dset, ds, (void *)base, n, dim, kDLFloat, 32, 0);
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = SK_GRAPH_DEGREE;
  h->ip->intermediate_graph_degree = SK_GRAPH_DEGREE;
  h->ip->nn_descent_niter = SK_NN_ITER;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { h->idx = 0; return -1; }
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) {
    cuvsCagraIndexDestroy(h->idx);
    h->idx = 0;
    return -1;
  }
  return 0;
}

/* --------------------------------------------------------------------------
 * Fast path.
 *
 * The kNN graph is computed on the GPU with IVF-Flat: cuVS keeps the lists in
 * row-major order, so every probe batch is a dense GEMM distance batch instead
 * of the scattered gather that NN-descent performs. On structured data
 * (e.g. SIFT-like vectors) this needs ~1/20 of the baseline build time.
 *
 * Because such a graph can be much weaker on data where clustering does not
 * reflect the neighbourhood structure, its quality is measured before it is
 * used: SK_SAMPLE rows are queried against the full dataset (full-probe IVF =
 * exact kNN) and the fraction of the exact 32-NN that the candidate graph
 * already contains has to reach SK_QCHECK_MIN. Two probe widths are tried; if
 * neither qualifies the caller falls back to NN-descent.
 * ------------------------------------------------------------------------ */
static int sk_build_ivf_graph(sk_cuvs_index *h, const float *base, long n, long dim)
{
  const long gd = SK_GRAPH_DEGREE;
  const long K = gd + 1;
  int rc = -1;
  cuvsIvfFlatIndexParams_t fip = 0;
  cuvsIvfFlatSearchParams_t fsp = 0;
  cuvsIvfFlatIndex_t fidx = 0;
  float *dx = 0, *dqs = 0;      /* device: dataset copy, sample queries   */
  float *dqc = 0, *ddc = 0;     /* device: chunk queries, chunk distances */
  unsigned short *dxh = 0;      /* device: FP16 dataset copy (dim>SK_F16_DIM) */
  unsigned short *dqs16 = 0;    /* device: FP16 sample queries            */
  unsigned short *dqc16 = 0;    /* device: FP16 chunk queries             */
  unsigned short *hch = 0;      /* host  : FP16 conversion scratch        */
  long long *dnc = 0;           /* device: chunk kNN ids                  */
  long long *hnc = 0;           /* host  : chunk kNN ids                  */
  unsigned *dgraph = 0;         /* device: final kNN graph                */
  unsigned *hgraph = 0;         /* host  : one graph chunk                */
  float *hq = 0;                /* host  : sample queries                 */
  long long *hex_h = 0, *hap_h = 0;
  long long *hex_d = 0, *hap_d = 0;
  float *hed_d = 0, *had_d = 0;
  int idx_ready = 0;
  const int use_f16 = (dim > SK_F16_DIM);

  if (use_f16) {
    size_t nv = (size_t)n * (size_t)dim;
    hch = (unsigned short *)malloc(2 * nv);
    if (hch == 0) goto done;
    for (size_t i = 0; i < nv; ++i) hch[i] = sk_f2h(base[i]);
    if (cuvsRMMAlloc(h->res, (void **)&dxh, 2 * nv) != CUVS_SUCCESS) goto done;
    if (cudaMemcpy(dxh, hch, 2 * nv, cudaMemcpyDefault) != cudaSuccess) goto done;
    free(hch); hch = 0;
  } else {
    if (cuvsRMMAlloc(h->res, (void **)&dx, sizeof(float) * (size_t)n * dim) != CUVS_SUCCESS) goto done;
    if (cudaMemcpy(dx, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault) != cudaSuccess) goto done;
  }

  if (cuvsIvfFlatIndexParamsCreate(&fip) != CUVS_SUCCESS) goto done;
  fip->metric = h->ip->metric;
  {
    /* n_lists ~ 2*sqrt(n): 250K rows -> 1024 lists (~250 rows per list). */
    long n_lists = 64;
    while (n_lists * n_lists < 4 * n && n_lists < 4096) { n_lists *= 2; }
    fip->n_lists = (uint32_t)n_lists;
  }
  fip->kmeans_n_iters = 10;
  fip->add_data_on_build = true;
  if (cuvsIvfFlatIndexCreate(&fidx) != CUVS_SUCCESS) goto done;
  {
    DLManagedTensor dset; long ds[2] = {n, dim};
    if (use_f16) sk_mk_tensor(&dset, ds, dxh, n, dim, kDLFloat, 16, 1);
    else          sk_mk_tensor(&dset, ds, dx,  n, dim, kDLFloat, 32, 1);
    if (cuvsIvfFlatBuild(h->res, fip, &dset, fidx) != CUVS_SUCCESS) goto done;
  }
  /* The IVF index holds its own copy of the data now; drop ours until the
   * CAGRA index needs it, which keeps the VRAM high-water mark at the IVF
   * phase instead of at the kNN phase. */
  if (dx != 0) { cuvsRMMFree(h->res, dx, sizeof(float) * (size_t)n * dim); dx = 0; }
  if (cuvsIvfFlatSearchParamsCreate(&fsp) != CUVS_SUCCESS) goto done;
  {
    uint32_t probes[2];
    int attempt, accepted_probes = 0;
    long step = n / SK_SAMPLE;
    long s;
    if (step < 1) step = 1;

    /* ---- sampled graph-quality probe ---- */
    hq = (float *)malloc(sizeof(float) * (size_t)SK_SAMPLE * dim);
    if (hq == 0) goto done;
    for (s = 0; s < SK_SAMPLE; ++s) {
      long row = s * step < n ? s * step : n - 1;
      memcpy(hq + (size_t)s * dim, base + (size_t)row * dim, sizeof(float) * dim);
    }
    if (use_f16) {
      size_t nv = (size_t)SK_SAMPLE * (size_t)dim;
      hch = (unsigned short *)malloc(2 * nv);
      if (hch == 0) goto done;
      for (size_t i = 0; i < nv; ++i) hch[i] = sk_f2h(hq[i]);
      if (cuvsRMMAlloc(h->res, (void **)&dqs16, 2 * nv) != CUVS_SUCCESS) goto done;
      if (cudaMemcpy(dqs16, hch, 2 * nv, cudaMemcpyDefault) != cudaSuccess) goto done;
      free(hch); hch = 0;
    } else {
      if (cuvsRMMAlloc(h->res, (void **)&dqs, sizeof(float) * (size_t)SK_SAMPLE * dim) != CUVS_SUCCESS) goto done;
      if (cudaMemcpy(dqs, hq, sizeof(float) * (size_t)SK_SAMPLE * dim, cudaMemcpyDefault) != cudaSuccess) goto done;
    }
    if (cuvsRMMAlloc(h->res, (void **)&hex_d, sizeof(long long) * (size_t)SK_SAMPLE * SK_EXACT_K) != CUVS_SUCCESS) goto done;
    if (cuvsRMMAlloc(h->res, (void **)&hed_d, sizeof(float) * (size_t)SK_SAMPLE * SK_EXACT_K) != CUVS_SUCCESS) goto done;
    if (cuvsRMMAlloc(h->res, (void **)&hap_d, sizeof(long long) * (size_t)SK_SAMPLE * gd) != CUVS_SUCCESS) goto done;
    if (cuvsRMMAlloc(h->res, (void **)&had_d, sizeof(float) * (size_t)SK_SAMPLE * gd) != CUVS_SUCCESS) goto done;
    hex_h = (long long *)malloc(sizeof(long long) * (size_t)SK_SAMPLE * SK_EXACT_K);
    hap_h = (long long *)malloc(sizeof(long long) * (size_t)SK_SAMPLE * gd);
    if (hex_h == 0 || hap_h == 0) goto done;
    {
      DLManagedTensor qt, nt, dt, n2, d2;
      long as[2], bs[2], cs[2], es[2], fs[2];
      cuvsFilter nf;
      nf.type = NO_FILTER; nf.addr = (uintptr_t)0;
      if (use_f16) sk_mk_tensor(&qt, as, dqs16, SK_SAMPLE, dim, kDLFloat, 16, 1);
      else          sk_mk_tensor(&qt, as, dqs, SK_SAMPLE, dim, kDLFloat, 32, 1);
      sk_mk_tensor(&nt, bs, hex_d, SK_SAMPLE, SK_EXACT_K, kDLInt, 64, 1);
      sk_mk_tensor(&dt, cs, hed_d, SK_SAMPLE, SK_EXACT_K, kDLFloat, 32, 1);
      sk_mk_tensor(&n2, es, hap_d, SK_SAMPLE, gd, kDLInt, 64, 1);
      sk_mk_tensor(&d2, fs, had_d, SK_SAMPLE, gd, kDLFloat, 32, 1);
      fsp->n_probes = fip->n_lists;   /* full probe == exact kNN */
      if (cuvsIvfFlatSearch(h->res, fsp, fidx, &qt, &nt, &dt, nf) != CUVS_SUCCESS) goto done;
      if (cudaMemcpy(hex_h, hex_d, sizeof(long long) * (size_t)SK_SAMPLE * SK_EXACT_K,
                     cudaMemcpyDefault) != cudaSuccess) goto done;
      probes[0] = (uint32_t)(fip->n_lists / 32 > 0 ? fip->n_lists / 32 : 1);
      probes[1] = (uint32_t)(fip->n_lists / 8 > 0 ? fip->n_lists / 8 : 1);
      for (attempt = 0; attempt < 2 && accepted_probes == 0; ++attempt) {
        long hit = 0, tot = 0;
        fsp->n_probes = probes[attempt];
        if (cuvsIvfFlatSearch(h->res, fsp, fidx, &qt, &n2, &d2, nf) != CUVS_SUCCESS) goto done;
        if (cudaMemcpy(hap_h, hap_d, sizeof(long long) * (size_t)SK_SAMPLE * gd,
                       cudaMemcpyDefault) != cudaSuccess) goto done;
        for (s = 0; s < SK_SAMPLE; ++s) {
          long k, j;
          for (k = 0; k < SK_EXACT_K; ++k) {
            long long v = hex_h[s * SK_EXACT_K + k];
            if (v < 0 || v == s * step) { continue; }
            ++tot;
            for (j = 0; j < gd; ++j) {
              if (hap_h[s * gd + j] == v) { ++hit; break; }
            }
          }
        }
        if (tot > 0 && (double)hit / (double)tot >= SK_QCHECK_MIN) { accepted_probes = (int)probes[attempt]; }
      }
      if (accepted_probes == 0) goto done;   /* too weak -> NN-descent */
      fsp->n_probes = (uint32_t)accepted_probes;
    }
    /* release the probe buffers */
    free(hex_h); hex_h = 0;
    free(hap_h); hap_h = 0;
    cuvsRMMFree(h->res, hex_d, sizeof(long long) * (size_t)SK_SAMPLE * SK_EXACT_K); hex_d = 0;
    cuvsRMMFree(h->res, hed_d, sizeof(float) * (size_t)SK_SAMPLE * SK_EXACT_K); hed_d = 0;
    cuvsRMMFree(h->res, hap_d, sizeof(long long) * (size_t)SK_SAMPLE * gd); hap_d = 0;
    cuvsRMMFree(h->res, had_d, sizeof(float) * (size_t)SK_SAMPLE * gd); had_d = 0;
    cuvsRMMFree(h->res, dqs, sizeof(float) * (size_t)SK_SAMPLE * dim); dqs = 0;
    if (dqs16 != 0) {
      cuvsRMMFree(h->res, dqs16, 2 * (size_t)SK_SAMPLE * dim); dqs16 = 0;
    }
    free(hq); hq = 0;
  }

  /* ---- full self-kNN, chunked, written straight into the device graph ---- */
  if (cuvsRMMAlloc(h->res, (void **)&dgraph, sizeof(unsigned) * (size_t)n * gd) != CUVS_SUCCESS) goto done;
  if (use_f16) {
    if (cuvsRMMAlloc(h->res, (void **)&dqc16, 2 * (size_t)SK_CHUNK * dim) != CUVS_SUCCESS) goto done;
    hch = (unsigned short *)malloc(2 * (size_t)SK_CHUNK * dim);
    if (hch == 0) goto done;
  } else {
    if (cuvsRMMAlloc(h->res, (void **)&dqc, sizeof(float) * (size_t)SK_CHUNK * dim) != CUVS_SUCCESS) goto done;
  }
  if (cuvsRMMAlloc(h->res, (void **)&dnc, sizeof(long long) * (size_t)SK_CHUNK * K) != CUVS_SUCCESS) goto done;
  if (cuvsRMMAlloc(h->res, (void **)&ddc, sizeof(float) * (size_t)SK_CHUNK * K) != CUVS_SUCCESS) goto done;
  hnc = (long long *)malloc(sizeof(long long) * (size_t)SK_CHUNK * K);
  hgraph = (unsigned *)malloc(sizeof(unsigned) * (size_t)SK_CHUNK * gd);
  if (hnc == 0 || hgraph == 0) goto done;
  {
    long start;
    for (start = 0; start < n; start += SK_CHUNK) {
      long rows = (n - start) < SK_CHUNK ? (n - start) : SK_CHUNK;
      long r;
      DLManagedTensor qt, nt, dt;
      long as[2], bs[2], cs[2];
      cuvsFilter nf;
      nf.type = NO_FILTER; nf.addr = (uintptr_t)0;
      if (use_f16) {
        const float *src = base + (size_t)start * dim;
        size_t nv = (size_t)rows * (size_t)dim;
        for (size_t i = 0; i < nv; ++i) hch[i] = sk_f2h(src[i]);
        if (cudaMemcpy(dqc16, hch, 2 * nv, cudaMemcpyDefault) != cudaSuccess) goto done;
        sk_mk_tensor(&qt, as, dqc16, rows, dim, kDLFloat, 16, 1);
      } else {
        if (cudaMemcpy(dqc, base + (size_t)start * dim, sizeof(float) * (size_t)rows * dim,
                       cudaMemcpyDefault) != cudaSuccess) goto done;
        sk_mk_tensor(&qt, as, dqc, rows, dim, kDLFloat, 32, 1);
      }
      sk_mk_tensor(&nt, bs, dnc, rows, K, kDLInt, 64, 1);
      sk_mk_tensor(&dt, cs, ddc, rows, K, kDLFloat, 32, 1);
      if (cuvsIvfFlatSearch(h->res, fsp, fidx, &qt, &nt, &dt, nf) != CUVS_SUCCESS) goto done;
      if (cudaMemcpy(hnc, dnc, sizeof(long long) * (size_t)rows * K, cudaMemcpyDefault) != cudaSuccess) goto done;
      for (r = 0; r < rows; ++r) {
        long w = 0, j;
        unsigned *grow = hgraph + (size_t)r * gd;
        long long self = start + r;
        for (j = 0; j < K && w < gd; ++j) {
          long long v = hnc[r * K + j];
          if (v < 0 || v == self || v >= n) { continue; }
          grow[w++] = (unsigned)v;
        }
        while (w < gd) { grow[w++] = (unsigned)((self + w + 1) % n); }
      }
      if (cudaMemcpy((char *)dgraph + (size_t)start * gd * sizeof(unsigned), hgraph,
                     sizeof(unsigned) * (size_t)rows * gd, cudaMemcpyDefault) != cudaSuccess) goto done;
    }
  }
  free(hnc); hnc = 0;
  free(hgraph); hgraph = 0;
  if (hch != 0) { free(hch); hch = 0; }
  cuvsRMMFree(h->res, ddc, sizeof(float) * (size_t)SK_CHUNK * K); ddc = 0;
  cuvsRMMFree(h->res, dnc, sizeof(long long) * (size_t)SK_CHUNK * K); dnc = 0;
  cuvsRMMFree(h->res, dqc, sizeof(float) * (size_t)SK_CHUNK * dim); dqc = 0;
  if (dqc16 != 0) { cuvsRMMFree(h->res, dqc16, 2 * (size_t)SK_CHUNK * dim); dqc16 = 0; }

  /* Drop the IVF index (it holds its own dataset copy) before materialising the
   * CAGRA index so the VRAM peak stays close to the baseline one. */
  cuvsIvfFlatSearchParamsDestroy(fsp); fsp = 0;
  cuvsIvfFlatIndexDestroy(fidx); fidx = 0;
  cuvsIvfFlatIndexParamsDestroy(fip); fip = 0;

  {
    DLManagedTensor gt, dt2;
    long as[2], bs[2];
    if (dx == 0) {
      if (cuvsRMMAlloc(h->res, (void **)&dx, sizeof(float) * (size_t)n * dim) != CUVS_SUCCESS) goto done;
      if (cudaMemcpy(dx, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault) != cudaSuccess) goto done;
    }
    sk_mk_tensor(&gt, as, dgraph, n, gd, kDLUInt, 32, 1);
    if (use_f16) sk_mk_tensor(&dt2, bs, dxh, n, dim, kDLFloat, 16, 1);
    else          sk_mk_tensor(&dt2, bs, dx,  n, dim, kDLFloat, 32, 1);
    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { h->idx = 0; goto done; }
    idx_ready = 1;
    if (cuvsCagraIndexFromArgs(h->res, h->ip->metric, &gt, &dt2, h->idx) != CUVS_SUCCESS) goto done;
  }
  /* Hand the construction buffers over to the handle: the index keeps using
   * them, so they are released in seekdb_cuvs_free(). */
  h->graph_buf = dgraph; dgraph = 0;
  h->f16 = use_f16;
  if (use_f16) { h->data_buf = dxh; dxh = 0; }
  else         { h->data_buf = dx;  dx = 0; }
  rc = 0;

done:
  if (rc != 0 && idx_ready) { cuvsCagraIndexDestroy(h->idx); h->idx = 0; }
  if (hex_h != 0) { free(hex_h); }
  if (hap_h != 0) { free(hap_h); }
  if (hq != 0) { free(hq); }
  if (hnc != 0) { free(hnc); }
  if (hgraph != 0) { free(hgraph); }
  if (fsp != 0) { cuvsIvfFlatSearchParamsDestroy(fsp); }
  if (fidx != 0) { cuvsIvfFlatIndexDestroy(fidx); }
  if (fip != 0) { cuvsIvfFlatIndexParamsDestroy(fip); }
  if (hex_d != 0) { cuvsRMMFree(h->res, hex_d, sizeof(long long) * (size_t)SK_SAMPLE * SK_EXACT_K); }
  if (hed_d != 0) { cuvsRMMFree(h->res, hed_d, sizeof(float) * (size_t)SK_SAMPLE * SK_EXACT_K); }
  if (hap_d != 0) { cuvsRMMFree(h->res, hap_d, sizeof(long long) * (size_t)SK_SAMPLE * gd); }
  if (had_d != 0) { cuvsRMMFree(h->res, had_d, sizeof(float) * (size_t)SK_SAMPLE * gd); }
  if (dqs != 0) { cuvsRMMFree(h->res, dqs, sizeof(float) * (size_t)SK_SAMPLE * dim); }
  if (dqc != 0) { cuvsRMMFree(h->res, dqc, sizeof(float) * (size_t)SK_CHUNK * dim); }
  if (dnc != 0) { cuvsRMMFree(h->res, dnc, sizeof(long long) * (size_t)SK_CHUNK * K); }
  if (ddc != 0) { cuvsRMMFree(h->res, ddc, sizeof(float) * (size_t)SK_CHUNK * K); }
  if (dgraph != 0) { cuvsRMMFree(h->res, dgraph, sizeof(unsigned) * (size_t)n * gd); }
  if (dx != 0) { cuvsRMMFree(h->res, dx, sizeof(float) * (size_t)n * dim); }
  if (dxh != 0) { cuvsRMMFree(h->res, dxh, 2 * (size_t)n * dim); }
  if (dqs16 != 0) { cuvsRMMFree(h->res, dqs16, 2 * (size_t)SK_SAMPLE * dim); }
  if (dqc16 != 0) { cuvsRMMFree(h->res, dqc16, 2 * (size_t)SK_CHUNK * dim); }
  if (hch != 0) { free(hch); }
  return rc;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;

  if (n >= SK_IVF_MIN_N && dim > 0 && dim <= SK_IVF_MAX_DIM &&
      sk_build_ivf_graph(h, base, n, dim) == 0) {
    h->n = n; h->dim = dim;
    return h;
  }
  if (sk_build_nn_descent(h, base, n, dim) == 0) {
    h->n = n; h->dim = dim;
    return h;
  }

  cuvsCagraIndexParamsDestroy(h->ip);
fail_res:
  cuvsResourcesDestroy(h->res);
  free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  unsigned short* qd16 = 0; unsigned short* qh16 = 0;
  const size_t qb = h->f16 ? 2 : 4;
  if (cuvsRMMAlloc(h->res, (void**)(h->f16 ? (void**)&qd16 : (void**)&qd), qb*(size_t)nq*h->dim) != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned)*nq*topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float)*nq*topk)    != CUVS_SUCCESS) { rc=12; goto done; }
  if (h->f16) {
    size_t nv = (size_t)nq * (size_t)h->dim, i;
    qh16 = (unsigned short*)malloc(2 * nv);
    if (qh16 == 0) { rc=13; goto done; }
    for (i = 0; i < nv; ++i) qh16[i] = sk_f2h(query[i]);
    if (cudaMemcpy(qd16, qh16, 2 * nv, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
    free(qh16); qh16 = 0;
  } else {
    if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data = h->f16 ? (void*)qd16 : (void*)qd;
    qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat;
    qt.dl_tensor.dtype.bits = h->f16 ? 16 : 32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (qh16) free(qh16);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float)*nq*h->dim);
  if (qd16) cuvsRMMFree(h->res, qd16, 2*(size_t)nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  if (h->graph_buf != 0) {
    cuvsRMMFree(h->res, h->graph_buf, sizeof(unsigned) * (size_t)h->n * SK_GRAPH_DEGREE);
    h->graph_buf = 0;
  }
  if (h->data_buf != 0) {
    cuvsRMMFree(h->res, h->data_buf,
                (h->f16 ? 2 : 4) * (size_t)h->n * (size_t)h->dim);
    h->data_buf = 0;
  }
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
