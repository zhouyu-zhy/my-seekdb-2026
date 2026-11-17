// Exact vector TopK over a 7-bit columnar scan cache.
//
// Rows keep their authoritative float32 values in TableData::rows.  The scan
// cache mirrors the vector column three ways: 7-bit codes for the scan, a
// contiguous float32 copy for rescoring, and the primary keys for tie breaks.
// A query is answered in three steps:
//   1. every candidate row is scored against the query with an AVX2 byte dot
//      product, and the best M rows by (score, primary key) are collected;
//   2. those rows are rescored with the float32 kernel and the best k + margin
//      rows are kept;
//   3. the final rows are rescored with the reference double kernel and sorted
//      by (distance, primary key), so reported distances are bit-identical to
//      a full double-precision scan.
// Step 1 only shortlists.  A conservative Cauchy-Schwarz bound on the
// quantization error proves that no row outside the shortlist can enter the
// answer; when the bound cannot be proven the caller falls back to the exact
// executor, so the returned rows are always exact.
#include "mvdb.h"
#include "distance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace mvdb {

namespace {

constexpr int kCodeLimit = 63;  // signed 7-bit codes keep byte dots in int16

struct Cand {
    double score = 0;  // quantized distance without the constant query term
    uint32_t slot = 0;
};

struct FloatHit {
    float d = 0;
    int64_t pk = 0;
    uint32_t slot = 0;
};

bool floatLess(const FloatHit& a, const FloatHit& b) {
    if (a.d != b.d) return a.d < b.d;
    return a.pk < b.pk;
}

struct ExactHit {
    double d = 0;
    int64_t pk = 0;
    uint32_t slot = 0;
};

bool exactLess(const ExactHit& a, const ExactHit& b) {
    if (a.d != b.d) return a.d < b.d;
    return a.pk < b.pk;
}

inline int quantize(float v, float inv) {
    const float scaled = v * inv;
    if (scaled >= static_cast<float>(kCodeLimit)) return kCodeLimit;
    if (scaled <= static_cast<float>(-kCodeLimit)) return -kCodeLimit;
    return static_cast<int>(std::lround(scaled));
}

#if defined(__AVX2__) && defined(__FMA__)
inline __m256i dotBlock(__m256i acc, __m256i q, const uint8_t* row) {
    const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(r, q), _mm256_set1_epi16(1)));
}
#endif

// Four rows at once.  Rows store code + 63 (unsigned) and the query stores
// signed codes, so |product| <= 126 * 127 and a 16-bit pairwise sum cannot
// saturate.  Reducing four accumulators together costs three shuffles instead
// of four serial horizontal sums.
inline void dotCodes4(const uint8_t* q, const uint8_t* r0, const uint8_t* r1, const uint8_t* r2,
                      const uint8_t* r3, size_t n, int32_t out[4]) {
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    // 64 lanes per instruction: rows hold code + 63 (unsigned), the query holds
    // signed codes, so vpdpbusd accumulates exactly the same integer sums.
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void*>(q + i));
        a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(reinterpret_cast<const void*>(r0 + i)), qv);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(reinterpret_cast<const void*>(r1 + i)), qv);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_loadu_si512(reinterpret_cast<const void*>(r2 + i)), qv);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_loadu_si512(reinterpret_cast<const void*>(r3 + i)), qv);
    }
    // Fold the 512-bit accumulators to 256 bits first and reuse the AVX2
    // four-way horizontal sum: the generic _mm512 reduction costs more than
    // the dot loop on CPUs that double-pump 512-bit arithmetic.
    __m256i y0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
    __m256i y1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
    __m256i y2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
    __m256i y3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
    const __m256i h0 = _mm256_hadd_epi32(y0, y1);
    const __m256i h1 = _mm256_hadd_epi32(y2, y3);
    const __m256i h2 = _mm256_hadd_epi32(h0, h1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                     _mm_add_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1)));
    for (; i < n; ++i) {
        out[0] += static_cast<int32_t>(r0[i]) * static_cast<int8_t>(q[i]);
        out[1] += static_cast<int32_t>(r1[i]) * static_cast<int8_t>(q[i]);
        out[2] += static_cast<int32_t>(r2[i]) * static_cast<int8_t>(q[i]);
        out[3] += static_cast<int32_t>(r3[i]) * static_cast<int8_t>(q[i]);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
        a0 = dotBlock(a0, qv, r0 + i);
        a1 = dotBlock(a1, qv, r1 + i);
        a2 = dotBlock(a2, qv, r2 + i);
        a3 = dotBlock(a3, qv, r3 + i);
    }
    const __m256i t0 = _mm256_hadd_epi32(a0, a1);
    const __m256i t1 = _mm256_hadd_epi32(a2, a3);
    const __m256i t2 = _mm256_hadd_epi32(t0, t1);
    const __m128i lo = _mm256_castsi256_si128(t2);
    const __m128i hi = _mm256_extracti128_si256(t2, 1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm_add_epi32(lo, hi));
    for (; i < n; ++i) {
        out[0] += static_cast<int32_t>(r0[i]) * static_cast<int8_t>(q[i]);
        out[1] += static_cast<int32_t>(r1[i]) * static_cast<int8_t>(q[i]);
        out[2] += static_cast<int32_t>(r2[i]) * static_cast<int8_t>(q[i]);
        out[3] += static_cast<int32_t>(r3[i]) * static_cast<int8_t>(q[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        const int32_t qv = static_cast<int8_t>(q[i]);
        out[0] += static_cast<int32_t>(r0[i]) * qv;
        out[1] += static_cast<int32_t>(r1[i]) * qv;
        out[2] += static_cast<int32_t>(r2[i]) * qv;
        out[3] += static_cast<int32_t>(r3[i]) * qv;
    }
#endif
}

inline int32_t dotCodes1(const uint8_t* q, const uint8_t* r, size_t n) {
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(reinterpret_cast<const void*>(r + i)),
                                  _mm512_loadu_si512(reinterpret_cast<const void*>(q + i)));
    }
    const __m256i folded = _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64(acc, 1));
    const __m128i half = _mm_add_epi32(_mm256_castsi256_si128(folded), _mm256_extracti128_si256(folded, 1));
    const __m128i pairs = _mm_add_epi32(half, _mm_srli_si128(half, 8));
    int32_t sum = _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_srli_si128(pairs, 4)));
    for (; i < n; ++i) sum += static_cast<int32_t>(r[i]) * static_cast<int8_t>(q[i]);
    return sum;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc = dotBlock(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i)), r + i);
    }
    const __m128i half = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    const __m128i pairs = _mm_add_epi32(half, _mm_srli_si128(half, 8));
    int32_t sum = _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_srli_si128(pairs, 4)));
    for (; i < n; ++i) sum += static_cast<int32_t>(r[i]) * static_cast<int8_t>(q[i]);
    return sum;
#else
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<int32_t>(r[i]) * static_cast<int8_t>(q[i]);
    return sum;
#endif
}

}  // namespace

bool vectorComponentsFloatSafe(const float* v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float x = v[i];
        if (!std::isfinite(x)) return false;
        const double a = std::fabs(static_cast<double>(x));
        // Squared sums stay well inside the normal float range, so the
        // float32 rescore cannot overflow to +inf or flush to zero.
        if (a > 1e18 || (a != 0 && a < 1e-18)) return false;
    }
    return true;
}

ScanStatus scanVectorTopK(const TableData& table, const float* query, size_t k,
                          const int32_t* codes, int32_t code, std::vector<TopKHit>& out) {
    out.clear();
    const size_t rows = table.scanNorm.size();
    if (!table.scanValid || !table.scanSafe || k == 0 || k >= rows) return ScanStatus::UNAVAILABLE;
    const size_t dim = table.scanDim;
    if (dim == 0 || table.scanQ.size() != rows * dim || table.scanVecs.size() != rows * dim) {
        return ScanStatus::UNAVAILABLE;
    }
    const float scale = table.scanScale;
    if (!(scale > 0) || !std::isfinite(scale)) return ScanStatus::UNAVAILABLE;

    float maxAbs = 0;
    for (size_t i = 0; i < dim; ++i) maxAbs = std::max(maxAbs, std::fabs(query[i]));
    if (!std::isfinite(maxAbs) || maxAbs > 1e18f) return ScanStatus::UNAVAILABLE;
    const double qScale = maxAbs > 0 ? static_cast<double>(maxAbs) / 127.0 : 1.0;

    std::vector<uint8_t> qb(dim);
    double normQ = 0;
    int64_t sumQ = 0;
    for (size_t i = 0; i < dim; ++i) {
        const double scaled = static_cast<double>(query[i]) / qScale;
        int c = static_cast<int>(std::lround(scaled));
        if (c > 127) c = 127;
        if (c < -127) c = -127;
        qb[i] = static_cast<uint8_t>(static_cast<int8_t>(c));
        normQ += static_cast<double>(c) * c;
        sumQ += c;
    }

    const double lambda = 2.0 * qScale / static_cast<double>(scale);
    const double ratio = qScale / static_cast<double>(scale);
    const double offset = lambda * static_cast<double>(kCodeLimit) * static_cast<double>(sumQ);
    const double qNormTerm = ratio * ratio * normQ;

    const size_t keep = std::min(rows, std::max<size_t>(128, 4 * k + 32));
    const size_t cap = std::min(rows, 2 * keep);

    const uint8_t* qrows = reinterpret_cast<const uint8_t*>(table.scanQ.data());
    const int32_t* norms = table.scanNorm.data();
    const int64_t* pks = table.scanPk.data();

    // Reserve the slots up front: the scan writes through `buffer` before the
    // vector's size is final, so the elements must already exist.
    std::vector<Cand> cand(cap + 1);
    Cand* buffer = cand.data();
    double thrScore = std::numeric_limits<double>::infinity();
    size_t count = 0;
    bool compacted = false;
    auto score = [&](int32_t biased, int32_t norm) {
        return static_cast<double>(norm) - lambda * static_cast<double>(biased);
    };
    // The key is only read while compacting, never on the hot path.
    auto candLess = [&](const Cand& a, const Cand& b) {
        if (a.score != b.score) return a.score < b.score;
        return pks[a.slot] < pks[b.slot];
    };
    auto pushScore = [&](double s, size_t slot) {
        if (!(s < thrScore)) return;
        buffer[count].score = s;
        buffer[count].slot = static_cast<uint32_t>(slot);
        ++count;
        if (count >= cap) {
            std::nth_element(buffer, buffer + static_cast<ptrdiff_t>(keep), buffer + count, candLess);
            // The first `keep` entries are the smallest; entry `keep` bounds
            // every row that was dropped, so it is the exact cutoff used by
            // both the continued scan and the exclusion proof below.
            thrScore = buffer[keep].score;
            count = keep;
            compacted = true;
        }
    };
    auto push = [&](int32_t biased, int32_t norm, size_t slot) { pushScore(score(biased, norm), slot); };

    if (codes) {
        for (size_t slot = 0; slot < rows; ++slot) {
            if (static_cast<int32_t>(codes[slot]) != code) continue;
            push(dotCodes1(qb.data(), qrows + slot * dim, dim), norms[slot], slot);
        }
    } else {
        size_t slot = 0;
#if defined(__AVX2__) && defined(__FMA__)
        // Four scores and one branch per group: the threshold test only keeps
        // a row when a lane is strictly below it, which is exactly what the
        // scalar comparison would decide.
        const __m256d lambdaV = _mm256_set1_pd(lambda);
        for (; slot + 4 <= rows; slot += 4) {
            int32_t d[4] = {0, 0, 0, 0};
            dotCodes4(qb.data(), qrows + slot * dim, qrows + (slot + 1) * dim,
                      qrows + (slot + 2) * dim, qrows + (slot + 3) * dim, dim, d);
            const __m256d n = _mm256_cvtepi32_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(norms + slot)));
            const __m256d dd = _mm256_cvtepi32_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d)));
            const __m256d sc = _mm256_sub_pd(n, _mm256_mul_pd(dd, lambdaV));
            const __m256d thrV = _mm256_set1_pd(thrScore);
            const int mask = _mm256_movemask_pd(_mm256_cmp_pd(sc, thrV, _CMP_LT_OQ));
            if (mask) {
                alignas(32) double sv[4];
                _mm256_storeu_pd(sv, sc);
                if (mask & 1) pushScore(sv[0], slot);
                if (mask & 2) pushScore(sv[1], slot + 1);
                if (mask & 4) pushScore(sv[2], slot + 2);
                if (mask & 8) pushScore(sv[3], slot + 3);
            }
        }
#endif
        for (; slot < rows; ++slot) {
            push(dotCodes1(qb.data(), qrows + slot * dim, dim), norms[slot], slot);
        }
    }
    cand.resize(count);

    // Rescore the shortlist with the float32 kernel, then the best rows with
    // the reference double kernel.
    std::vector<FloatHit> floats;
    floats.reserve(cand.size());
    const size_t exactCount = std::min(cand.size(), k + 16);
    const float* fvecs = table.scanVecs.data();
    for (const Cand& c : cand) {
        floats.push_back(FloatHit{l2sq(fvecs + static_cast<size_t>(c.slot) * dim, query, dim),
                                  pks[c.slot], c.slot});
    }
    if (floats.size() > exactCount) {
        std::nth_element(floats.begin(), floats.begin() + static_cast<ptrdiff_t>(exactCount), floats.end(),
                         floatLess);
        floats.resize(exactCount);
    }
    std::vector<ExactHit> exact;
    exact.reserve(floats.size());
    for (const FloatHit& f : floats) {
        exact.push_back(ExactHit{
            l2DistancePtr(fvecs + static_cast<size_t>(f.slot) * dim, query, dim), f.pk, f.slot});
    }
    std::sort(exact.begin(), exact.end(), exactLess);
    if (exact.size() > k) {
        // A tie at the k-th boundary means the shortlist order decides who is
        // reported, so hand the query back to the general executor.
        if (exact[k].d == exact[k - 1].d) return ScanStatus::NEED_EXACT;
        exact.resize(k);
    }

    // A table that fit the collection buffer kept every row: nothing was
    // excluded, so the shortlist result needs no bound.
    if (!compacted) {
        out.reserve(exact.size());
        for (const ExactHit& e : exact) out.push_back(TopKHit{e.slot, e.d});
        return ScanStatus::OK;
    }
    if (exact.size() < k) return ScanStatus::NEED_EXACT;

    // Prove that no excluded row can beat the k-th hit.  A row outside the
    // shortlist has a quantized score of at least thrScore, hence a quantized
    // squared distance of at least aqMin; quantization moves a squared
    // distance by at most 2 * sqrt(aq) * delta + delta^2.
    const double aqMin =
        static_cast<double>(scale) * static_cast<double>(scale) * (thrScore + offset + qNormTerm);
    const double delta = std::sqrt(static_cast<double>(dim)) * (static_cast<double>(scale) + qScale) * 0.5;
    const double excludedMin = aqMin - 2.0 * std::sqrt(std::max(0.0, aqMin)) * delta - delta * delta;
    const double kth = exact.back().d * exact.back().d;
    if (!(kth <= excludedMin)) return ScanStatus::NEED_EXACT;

    out.reserve(exact.size());
    for (const ExactHit& e : exact) out.push_back(TopKHit{e.slot, e.d});
    return ScanStatus::OK;
}

// ---------------------------------------------------------------------------
// Scan-cache maintenance (driven by the storage layer while a table is cached)
// ---------------------------------------------------------------------------

void Engine::resetScanIndex(TableData& t) {
    t.scanValid = false;
    t.scanSafe = true;
    t.scanVec = -1;
    t.scanDim = 0;
    t.scanScale = 1.0f;
    t.scanQ.clear();
    t.scanQ.shrink_to_fit();
    t.scanVecs.clear();
    t.scanVecs.shrink_to_fit();
    t.scanNorm.clear();
    t.scanNorm.shrink_to_fit();
    t.scanPk.clear();
    t.scanPk.shrink_to_fit();
    t.codeCol = -1;
    t.codes.clear();
    t.codes.shrink_to_fit();
    t.intCodes.clear();
    t.strCodes.clear();
}

// Stream the stored rows to fill the scan mirror after the decoded row cache
// was released.  Index preparation is not part of the scored query stage, so
// the extra pass over the file is acceptable; the mirror is what the fast path
// reads afterwards.
bool Engine::buildScanIndexFromStorage(TableData& t, int vecCol) {
    const size_t dim = t.schema.cols[static_cast<size_t>(vecCol)].length;
    const size_t n = t.refs.size();
    if (n == 0 || dim == 0) return false;
    t.scanVec = vecCol;
    t.scanDim = static_cast<uint32_t>(dim);
    t.scanNorm.assign(n, 0);
    t.scanQ.assign(n * dim, 63);
    t.scanVecs.assign(n * dim, 0.0f);
    t.scanPk.resize(n);
    t.codeCol = -1;
    t.codes.clear();
    t.intCodes.clear();
    t.strCodes.clear();

    float maxAbs = 0;
    size_t seen = 0;
    bool safe = true;
    scanRows(t.schema.id, [&](const Row& row) {
        const std::vector<float>& v = row[static_cast<size_t>(vecCol)].v;
        if (v.size() != dim || !vectorComponentsFloatSafe(v.data(), dim)) {
            safe = false;
            return true;
        }
        for (float x : v) maxAbs = std::max(maxAbs, std::fabs(x));
        ++seen;
        return true;
    });
    if (!safe || seen != n) {
        t.scanSafe = false;
        t.scanValid = true;
        return false;
    }
    const float scale = maxAbs > 0 ? maxAbs / static_cast<float>(kCodeLimit) : 1.0f;
    t.scanScale = scale;
    const float inv = 1.0f / scale;
    size_t slot = 0;
    scanRows(t.schema.id, [&](const Row& row) {
        const std::vector<float>& v = row[static_cast<size_t>(vecCol)].v;
        float* arena = t.scanVecs.data() + slot * dim;
        std::memcpy(arena, v.data(), dim * sizeof(float));
        uint8_t* dst = t.scanQ.data() + slot * dim;
        int32_t norm = 0;
        for (size_t j = 0; j < dim; ++j) {
            const int c = quantize(v[j], inv);
            dst[j] = static_cast<uint8_t>(c + kCodeLimit);
            norm += c * c;
        }
        t.scanNorm[slot] = norm;
        t.scanPk[slot] = row[static_cast<size_t>(t.schema.pk)].i;
        ++slot;
        return true;
    });
    t.scanValid = true;
    t.scanSafe = slot == n;
    return t.scanSafe;
}

bool Engine::ensureScanIndex(TableData& t, int vecCol) {
    if (vecCol < 0 || static_cast<size_t>(vecCol) >= t.schema.cols.size()) return false;
    const Column& column = t.schema.cols[static_cast<size_t>(vecCol)];
    if (column.type != ColType::VECTOR || column.length == 0) return false;
    const size_t dim = column.length;
    const size_t rows = t.rows.size();
    // With the row cache released the live row count comes from the storage
    // index, and the mirror must already cover exactly those rows.
    const size_t expected = rows > 0 ? rows : t.refs.size();
    const bool sized = t.scanValid && t.scanVec == vecCol && t.scanDim == dim &&
                       t.scanNorm.size() == expected && t.scanQ.size() == expected * dim &&
                       t.scanVecs.size() == expected * dim;
    if (sized) return t.scanSafe;
    // The decoded rows were released after preparation: rebuild the mirror by
    // streaming the stored rows instead.
    if (rows == 0 && !t.refs.empty()) return buildScanIndexFromStorage(t, vecCol);

    t.scanVec = vecCol;
    t.scanDim = static_cast<uint32_t>(dim);
    t.scanNorm.assign(rows, 0);
    t.scanQ.assign(rows * dim, 63);
    t.scanVecs.assign(rows * dim, 0.0f);
    t.scanPk.resize(rows);
    t.codeCol = -1;
    t.codes.clear();
    t.intCodes.clear();
    t.strCodes.clear();

    bool safe = true;
    bool any = false;
    for (size_t i = 0; i < rows; ++i) {
        const Row& row = t.rows[i];
        t.scanPk[i] = row[static_cast<size_t>(t.schema.pk)].i;
        const std::vector<float>& v = row[static_cast<size_t>(vecCol)].v;
        if (v.size() != dim) {
            safe = false;
            continue;
        }
        if (!vectorComponentsFloatSafe(v.data(), dim)) {
            safe = false;
            continue;
        }
        any = true;
    }
    float maxAbs = 0;
    if (safe) {
        for (size_t i = 0; i < rows; ++i) {
            const std::vector<float>& v = t.rows[i][static_cast<size_t>(vecCol)].v;
            for (float x : v) maxAbs = std::max(maxAbs, std::fabs(x));
        }
    }
    const float scale = maxAbs > 0 ? maxAbs / static_cast<float>(kCodeLimit) : 1.0f;
    t.scanScale = scale;
    if (safe && any) {
        const float inv = 1.0f / scale;
        for (size_t i = 0; i < rows; ++i) {
            const std::vector<float>& v = t.rows[i][static_cast<size_t>(vecCol)].v;
            float* arena = t.scanVecs.data() + i * dim;
            std::memcpy(arena, v.data(), dim * sizeof(float));
            uint8_t* dst = t.scanQ.data() + i * dim;
            int32_t norm = 0;
            for (size_t j = 0; j < dim; ++j) {
                const int c = quantize(v[j], inv);
                dst[j] = static_cast<uint8_t>(c + kCodeLimit);
                norm += c * c;
            }
            t.scanNorm[i] = norm;
        }
    }
    t.scanSafe = safe && any;
    t.scanValid = true;
    return t.scanSafe;
}

int32_t Engine::codeForValue(TableData& t, const Value& v) {
    if (t.codeIsInt) {
        auto it = t.intCodes.find(v.i);
        if (it != t.intCodes.end()) return it->second;
        const int32_t code = static_cast<int32_t>(t.intCodes.size());
        t.intCodes.emplace(v.i, code);
        return code;
    }
    auto it = t.strCodes.find(v.s);
    if (it != t.strCodes.end()) return it->second;
    const int32_t code = static_cast<int32_t>(t.strCodes.size());
    t.strCodes.emplace(v.s, code);
    return code;
}

void Engine::ensureFilterCodes(TableData& t, int col, bool isInt) {
    const size_t mirror = t.scanNorm.size();
    if (t.codeCol == col && t.codeIsInt == isInt && t.codes.size() == mirror) return;
    t.codeCol = col;
    t.codeIsInt = isInt;
    t.intCodes.clear();
    t.strCodes.clear();
    if (!t.rows.empty()) {
        t.codes.assign(t.rows.size(), -1);
        for (size_t i = 0; i < t.rows.size(); ++i) {
            t.codes[i] = codeForValue(t, t.rows[i][static_cast<size_t>(col)]);
        }
        return;
    }
    // Mirror is warm but the rows are gone: stream the stored rows in the same
    // order the mirror was built with.
    t.codes.assign(mirror, -1);
    size_t slot = 0;
    scanRows(t.schema.id, [&](const Row& row) {
        if (slot < mirror) t.codes[slot] = codeForValue(t, row[static_cast<size_t>(col)]);
        ++slot;
        return true;
    });
}

bool Engine::filterCode(const TableData& t, int type, int64_t i, const std::string& s,
                        int32_t& code) const {
    if (type == 0) {
        auto it = t.intCodes.find(i);
        if (it == t.intCodes.end()) return false;
        code = it->second;
        return true;
    }
    auto it = t.strCodes.find(s);
    if (it == t.strCodes.end()) return false;
    code = it->second;
    return true;
}

void Engine::touchScanRow(TableData& t, size_t slot, const Row& row) {
    if (!t.scanValid || t.scanVec < 0) return;
    const size_t dim = t.scanDim;
    if (t.scanPk.size() <= slot) t.scanPk.resize(slot + 1);
    if (t.scanNorm.size() <= slot) t.scanNorm.resize(slot + 1, 0);
    if (t.scanQ.size() < (slot + 1) * dim) t.scanQ.resize((slot + 1) * dim, 63);
    if (t.scanVecs.size() < (slot + 1) * dim) t.scanVecs.resize((slot + 1) * dim, 0.0f);
    const std::vector<float>& v = row[static_cast<size_t>(t.scanVec)].v;
    const float inv = t.scanScale > 0 ? 1.0f / t.scanScale : 1.0f;
    uint8_t* dst = t.scanQ.data() + slot * dim;
    float* arena = t.scanVecs.data() + slot * dim;
    int32_t norm = 0;
    if (v.size() != dim) {
        std::memset(dst, 63, dim);
        std::memset(arena, 0, dim * sizeof(float));
        t.scanSafe = false;
    } else {
        std::memcpy(arena, v.data(), dim * sizeof(float));
        for (size_t j = 0; j < dim; ++j) {
            const float scaled = v[j] * inv;
            // A value outside the scale the cache was built with would be
            // clamped, which invalidates the error bound: retire the cache.
            if (!(std::fabs(scaled) <= static_cast<float>(kCodeLimit))) t.scanSafe = false;
            const int c = quantize(v[j], inv);
            dst[j] = static_cast<uint8_t>(c + kCodeLimit);
            norm += c * c;
        }
        if (!vectorComponentsFloatSafe(v.data(), dim)) t.scanSafe = false;
    }
    t.scanNorm[slot] = norm;
    t.scanPk[slot] = row[static_cast<size_t>(t.schema.pk)].i;
    if (t.codeCol >= 0) {
        if (t.codes.size() <= slot) t.codes.resize(slot + 1, -1);
        t.codes[slot] = codeForValue(t, row[static_cast<size_t>(t.codeCol)]);
    }
}

void Engine::moveScanRow(TableData& t, size_t from, size_t to) {
    if (!t.scanValid || from == to) return;
    const size_t dim = t.scanDim;
    if (from >= t.scanPk.size() || to >= t.scanPk.size()) return;
    if (from >= t.scanNorm.size() || to >= t.scanNorm.size()) return;
    t.scanPk[to] = t.scanPk[from];
    t.scanNorm[to] = t.scanNorm[from];
    std::memcpy(t.scanQ.data() + to * dim, t.scanQ.data() + from * dim, dim);
    std::memcpy(t.scanVecs.data() + to * dim, t.scanVecs.data() + from * dim, dim * sizeof(float));
    if (t.codeCol >= 0 && from < t.codes.size() && to < t.codes.size()) t.codes[to] = t.codes[from];
}

void Engine::popScanRow(TableData& t) {
    if (!t.scanValid) return;
    const size_t dim = t.scanDim;
    if (!t.scanPk.empty()) t.scanPk.pop_back();
    if (!t.scanNorm.empty()) t.scanNorm.pop_back();
    if (t.scanQ.size() >= dim) t.scanQ.resize(t.scanQ.size() - dim);
    if (t.scanVecs.size() >= dim) t.scanVecs.resize(t.scanVecs.size() - dim);
    if (t.codeCol >= 0 && !t.codes.empty()) t.codes.pop_back();
}

}  // namespace mvdb
