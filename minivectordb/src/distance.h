// Distance kernels shared by the row executor and the columnar TopK scan.
//
// The double kernel is the reference: results must stay bit-identical to the
// scalar accumulation order used since the first version, because the stored
// WAL and the public test expectations both depend on it.  The float kernel is
// only used to shortlist candidates; the shortlist is rescored with the double
// kernel before anything is returned.
#pragma once

#include <cmath>
#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mvdb {

inline double l2DistancePtr(const float* a, const float* b, size_t n) {
    size_t i = 0;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i + 4 <= n; i += 4) {
        double d0 = static_cast<double>(a[i]) - b[i];
        double d1 = static_cast<double>(a[i + 1]) - b[i + 1];
        double d2 = static_cast<double>(a[i + 2]) - b[i + 2];
        double d3 = static_cast<double>(a[i + 3]) - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    double s = s0 + s1 + s2 + s3;
    for (; i < n; ++i) {
        double d = static_cast<double>(a[i]) - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

inline float l2sqScalar(const float* a, const float* b, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#if defined(__AVX2__) && defined(__FMA__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 sum = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, sum);
    sum = _mm_add_ss(sum, sh);
    return _mm_cvtss_f32(sum);
}

// Squared float L2 distance of one row against a fixed query.
inline float l2sqAvx2(const float* a, const float* b, size_t n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    float sum = hsum256(_mm256_add_ps(s0, s1));
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Two rows at once: four independent FMA chains keep the pipeline busy while
// the query block is loaded once per pair.
inline void l2sqAvx2Pair(const float* a0, const float* a1, const float* b, size_t n,
                         float& out0, float& out1) {
    __m256 p0 = _mm256_setzero_ps(), p1 = _mm256_setzero_ps();
    __m256 q0 = _mm256_setzero_ps(), q1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 b0 = _mm256_loadu_ps(b + i);
        const __m256 b1 = _mm256_loadu_ps(b + i + 8);
        const __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a0 + i), b0);
        const __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a0 + i + 8), b1);
        const __m256 y0 = _mm256_sub_ps(_mm256_loadu_ps(a1 + i), b0);
        const __m256 y1 = _mm256_sub_ps(_mm256_loadu_ps(a1 + i + 8), b1);
        p0 = _mm256_fmadd_ps(x0, x0, p0);
        p1 = _mm256_fmadd_ps(x1, x1, p1);
        q0 = _mm256_fmadd_ps(y0, y0, q0);
        q1 = _mm256_fmadd_ps(y1, y1, q1);
    }
    out0 = hsum256(_mm256_add_ps(p0, p1));
    out1 = hsum256(_mm256_add_ps(q0, q1));
    for (; i < n; ++i) {
        const float d0 = a0[i] - b[i];
        const float d1 = a1[i] - b[i];
        out0 += d0 * d0;
        out1 += d1 * d1;
    }
}

inline float l2sq(const float* a, const float* b, size_t n) { return l2sqAvx2(a, b, n); }

inline void l2sqPair(const float* a0, const float* a1, const float* b, size_t n,
                     float& out0, float& out1) {
    l2sqAvx2Pair(a0, a1, b, n, out0, out1);
}
#else
inline float l2sq(const float* a, const float* b, size_t n) { return l2sqScalar(a, b, n); }

inline void l2sqPair(const float* a0, const float* a1, const float* b, size_t n,
                     float& out0, float& out1) {
    out0 = l2sqScalar(a0, b, n);
    out1 = l2sqScalar(a1, b, n);
}
#endif

}  // namespace mvdb
