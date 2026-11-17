#pragma once
#include <cstddef>
#include <string>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mvdb {

// Return the next quote, escape, or (for JSON) illegal control byte.
// A block is loaded only when it lies wholly inside the string.
inline size_t scanStringSpan(const std::string& text, size_t pos, char quote, bool json)
{
#if defined(__AVX2__)
    const __m256i quotes = _mm256_set1_epi8(quote);
    const __m256i escapes = _mm256_set1_epi8('\\');
    const __m256i controls = _mm256_set1_epi8(0x1f);
    for (; pos + 32 <= text.size(); pos += 32) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(text.data() + pos));
        __m256i special = _mm256_or_si256(_mm256_cmpeq_epi8(bytes, quotes),
                                        _mm256_cmpeq_epi8(bytes, escapes));
        if (json) {
            special = _mm256_or_si256(special,
                _mm256_cmpeq_epi8(_mm256_min_epu8(bytes, controls), bytes));
        }
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_epi8(special));
        if (mask) return pos + static_cast<size_t>(__builtin_ctz(mask));
    }
#endif
    while (pos < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[pos]);
        if (c == static_cast<unsigned char>(quote) || c == '\\' || (json && c < 0x20)) break;
        ++pos;
    }
    return pos;
}
}
