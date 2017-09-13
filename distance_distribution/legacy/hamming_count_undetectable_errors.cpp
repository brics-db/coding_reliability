// Copyright 2015,2016 Till Kolditz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <type_traits>
#include <omp.h>
#include <immintrin.h>
#include <boost/multiprecision/cpp_int.hpp>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt64
#endif

/**
 * Taken and adapted from: https://gist.github.com/jeetsukumaran/5392166
 *
 * Calculates the binomial coefficient, $\choose{n, k}$, i.e., the number of
 * distinct sets of $k$ elements that can be sampled with replacement from a
 * population of $n$ elements.
 *
 * @tparam T
 *   Numeric type. Defaults to unsigned long.
 * @param n
 *   Population size.
 * @param k
 *   Number of elements to sample without replacement.
 *
 * @return
 *   The binomial coefficient, $\choose{n, k}$.
 *
 * @note
 *    Modified from: http://etceterology.com/fast-binomial-coefficients
 */
template<class T = unsigned long>
T binomial_coefficient(
        size_t n,
        size_t k) {
    unsigned long i;
    if (0 == k || n == k) {
        return 1;
    }
    if (k > n) {
        return 0;
    }
    if (k > (n - k)) {
        k = n - k;
    }
    if (1 == k) {
        return n;
    }
    T b = 1;
    for (i = 1; i <= k; ++i) {
        b *= (n - (k - i));
        if (b < 0)
            return -1; /* Overflow */
        b /= i;
    }
    return b;
}

class StopWatch {
    std::chrono::high_resolution_clock::time_point startNS, stopNS;

public:
    StopWatch()
            : startNS(std::chrono::nanoseconds(0)),
              stopNS(std::chrono::nanoseconds(0)) {
    }

    void start() {
        startNS = std::chrono::high_resolution_clock::now();
    }

    std::chrono::high_resolution_clock::rep stop() {
        stopNS = std::chrono::high_resolution_clock::now();
        return duration();
    }

    std::chrono::high_resolution_clock::rep duration() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(stopNS - startNS).count();
    }
};
std::ostream& operator<<(
        std::ostream& stream,
        StopWatch& sw) {
    std::chrono::high_resolution_clock::rep dura = sw.duration();
    size_t max = 1000;
    while (dura / max > 0) {
        max *= 1000;
    }
    max /= 1000;
    stream << std::setfill('0') << (dura / max);
    while (max > 1) {
        dura %= max;
        max /= 1000;
        stream << '.' << std::setw(3) << (dura / max);
    }
    stream << std::setfill(' ');
    return stream;
}

/*
 * Count undetectable bit flips for Hamming encoded data words.
 *
 * Extended Hamming cannot detect bit flips which lead
 *  * either exactly to another codeword,
 *  * or to any other word in the 1-distance sphere around another codeword.
 *
 * We essentially only count all transitions between codewords and other codewords,
 * as well as codewords and non-codewords which are 1 bit away from the codewords (1-distance sphere),
 * because we know that these transistions cannot be detected.
 *
 * Therefore, we model the space of possible messages as a graph. The edges are the transitions / bit flips between words,
 * while the weights are the number of bits required to flip to get to the other message.
 * Since we start from codewords only, the graph is directed, from all codewords to all other codewords an all words in the
 * corresponding 1-distance sphears.
 *
 * The 1-distance sphere is only virtual in this implementation, realized by an inner for-loop over all 22 possible 1-bitflips
 *
 * The weights (W) are counted, grouped by the weight. This gives the corresponding number of undetectable W-bitflips.
 */

template<typename V, typename T>
struct SIMD;

template<typename T>
struct Scalar;

template<>
struct SIMD<__m128i, uint8_t> {
    static __m128i set1(
            uint8_t value) {
        return _mm_set1_epi8(value);
    }

    static __m128i set_inc(
            uint8_t base = 0,
            uint8_t inc = 1) {
        return _mm_set_epi8(static_cast<uint8_t>(base + 15 * inc), static_cast<uint8_t>(base + 14 * inc), static_cast<uint8_t>(base + 13 * inc), static_cast<uint8_t>(base + 12 * inc),
                static_cast<uint8_t>(base + 11 * inc), static_cast<uint8_t>(base + 10 * inc), static_cast<uint8_t>(base + 9 * inc), static_cast<uint8_t>(base + 8 * inc),
                static_cast<uint8_t>(base + 7 * inc), static_cast<uint8_t>(base + 6 * inc), static_cast<uint8_t>(base + 5 * inc), static_cast<uint8_t>(base + 4 * inc),
                static_cast<uint8_t>(base + 3 * inc), static_cast<uint8_t>(base + 2 * inc), static_cast<uint8_t>(base + 1 * inc), base);
    }

    static __m128i add(
            __m128i a,
            __m128i b) {
        return _mm_add_epi8(a, b);
    }

    static __m128i popcount(
            __m128i a) {
        auto lookup = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
        auto low_mask = _mm_set1_epi8(0x0f);
        auto lo = _mm_and_si128(a, low_mask);
        auto hi = _mm_and_si128(_mm_srli_epi16(a, 4), low_mask);
        auto cnt_lo = _mm_shuffle_epi8(lookup, lo);
        auto cnt_hi = _mm_shuffle_epi8(lookup, hi);
        return _mm_add_epi8(cnt_lo, cnt_hi);
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi8(static_cast<int8_t>(0x5B));
        auto pattern2 = _mm_set1_epi8(static_cast<int8_t>(0x6D));
        auto pattern3 = _mm_set1_epi8(static_cast<int8_t>(0x8E));
        auto pattern4 = _mm_set1_epi8(static_cast<int8_t>(0xF0));
        auto mask = _mm_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        auto codebits = _mm_slli_epi16(tmp2, shift);
        auto tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern3)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern4)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(_mm_add_epi8(popcount(data), tmp2), mask);
        codebits = _mm_or_si128(codebits, tmp1);
        return codebits;
    }
};

template<>
struct SIMD<__m128i, uint16_t> {
    static __m128i set1(
            uint16_t value) {
        return _mm_set1_epi16(value);
    }

    static __m128i set_inc(
            uint16_t base = 0,
            uint16_t inc = 1) {
        return _mm_set_epi16(static_cast<uint16_t>(base + 7 * inc), static_cast<uint16_t>(base + 6 * inc), static_cast<uint16_t>(base + 5 * inc), static_cast<uint16_t>(base + 4 * inc),
                static_cast<uint16_t>(base + 3 * inc), static_cast<uint16_t>(base + 2 * inc), static_cast<uint16_t>(base + 1 * inc), base);
    }

    static __m128i add(
            __m128i a,
            __m128i b) {
        return _mm_add_epi16(a, b);
    }

    static __m128i popcount(
            __m128i a) {
        auto mask = _mm_set1_epi8(0x01);
        auto shuffle = _mm_set_epi32(0xFF0FFF0D, 0xFF0BFF09, 0xFF07FF05, 0xFF03FF01);
        auto popcount8 = SIMD<__m128i, uint8_t>::popcount(a);
        return _mm_shuffle_epi8(_mm_mullo_epi16(popcount8, mask), shuffle);
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi16(static_cast<int16_t>(0xAD5B));
        auto pattern2 = _mm_set1_epi16(static_cast<int16_t>(0x366D));
        auto pattern3 = _mm_set1_epi16(static_cast<int16_t>(0xC78E));
        auto pattern4 = _mm_set1_epi16(static_cast<int16_t>(0x07F0));
        auto pattern5 = _mm_set1_epi16(static_cast<int16_t>(0xF800));
        auto mask = _mm_set1_epi16(static_cast<int16_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        auto codebits = _mm_slli_epi16(tmp2, shift);
        auto tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern3)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern4)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern5)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        codebits = _mm_or_si128(codebits, _mm_and_si128(_mm_add_epi16(popcount(data), tmp2), mask));
        return codebits;
    }
};

template<>
struct SIMD<__m128i, uint32_t> {
    static __m128i set1(
            uint32_t value) {
        return _mm_set1_epi32(value);
    }

    static __m128i set_inc(
            uint32_t base = 0,
            uint32_t inc = 1) {
        return _mm_set_epi32(static_cast<uint32_t>(base + 3 * inc), static_cast<uint32_t>(base + 2 * inc), static_cast<uint32_t>(base + 1 * inc), base);
    }

    static __m128i add(
            __m128i a,
            __m128i b) {
        return _mm_add_epi32(a, b);
    }

    static __m128i popcount(
            __m128i a) {
        auto mask = _mm_set1_epi8(0x01);
        auto shuffle = _mm_set_epi32(0xFFFFFF0F, 0xFFFFFF0B, 0xFFFFFF07, 0xFFFFFF03);
        auto popcount8 = SIMD<__m128i, uint8_t>::popcount(a);
        return _mm_shuffle_epi8(_mm_mullo_epi32(popcount8, mask), shuffle);
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi32(0x56AAAD5B);
        auto pattern2 = _mm_set1_epi32(0x9B33366D);
        auto pattern3 = _mm_set1_epi32(0xE3C3C78E);
        auto pattern4 = _mm_set1_epi32(0x03FC07F0);
        auto pattern5 = _mm_set1_epi32(0x03FFF800);
        auto pattern6 = _mm_set1_epi32(0xFC000000);
        auto mask = _mm_set1_epi32(0x1);
        uint8_t shift = 1;
        __m128i tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        __m128i codebits = _mm_slli_epi32(tmp2, shift);
        __m128i tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern3)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern4)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern5)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern6)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        codebits = _mm_or_si128(codebits, _mm_and_si128(_mm_add_epi32(popcount(data), tmp2), mask));
        return codebits;
    }
};

template<>
struct SIMD<__m128i, uint64_t> {
    static __m128i set1(
            uint64_t value) {
        return _mm_set1_epi64x(value);
    }

    static __m128i set_inc(
            uint64_t base = 0,
            uint64_t inc = 1) {
        return _mm_set_epi64x(base + inc, base);
    }

    static __m128i add(
            __m128i a,
            __m128i b) {
        return _mm_add_epi64(a, b);
    }

    static __m128i popcount(
            __m128i a) {
        return _mm_set_epi64x(__builtin_popcountll(_mm_extract_epi64(a, 1)), __builtin_popcountll(_mm_extract_epi64(a, 0)));
    }
};

#ifdef __AVX2__

template<>
struct SIMD<__m256i, uint8_t> {
    static __m256i set1(
            uint8_t value) {
        return _mm256_set1_epi8(value);
    }

    static __m256i set_inc(
            uint8_t base = 0,
            uint8_t inc = 1) {
        return _mm256_set_epi8(static_cast<uint8_t>(base + 31 * inc), static_cast<uint8_t>(base + 30 * inc), static_cast<uint8_t>(base + 29 * inc), static_cast<uint8_t>(base + 28 * inc),
                static_cast<uint8_t>(base + 27 * inc), static_cast<uint8_t>(base + 26 * inc), static_cast<uint8_t>(base + 25 * inc), static_cast<uint8_t>(base + 24 * inc),
                static_cast<uint8_t>(base + 23 * inc), static_cast<uint8_t>(base + 22 * inc), static_cast<uint8_t>(base + 21 * inc), static_cast<uint8_t>(base + 20 * inc),
                static_cast<uint8_t>(base + 19 * inc), static_cast<uint8_t>(base + 18 * inc), static_cast<uint8_t>(base + 17 * inc), static_cast<uint8_t>(base + 16 * inc),
                static_cast<uint8_t>(base + 15 * inc), static_cast<uint8_t>(base + 14 * inc), static_cast<uint8_t>(base + 13 * inc), static_cast<uint8_t>(base + 12 * inc),
                static_cast<uint8_t>(base + 11 * inc), static_cast<uint8_t>(base + 10 * inc), static_cast<uint8_t>(base + 9 * inc), static_cast<uint8_t>(base + 8 * inc),
                static_cast<uint8_t>(base + 7 * inc), static_cast<uint8_t>(base + 6 * inc), static_cast<uint8_t>(base + 5 * inc), static_cast<uint8_t>(base + 4 * inc),
                static_cast<uint8_t>(base + 3 * inc), static_cast<uint8_t>(base + 2 * inc), static_cast<uint8_t>(base + 1 * inc), base);
    }

    static __m256i add(
            __m256i a,
            __m256i b) {
        return _mm256_add_epi8(a, b);
    }

    static __m256i popcount(
            __m256i a) {
        auto lookup = _mm256_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
        auto low_mask = _mm256_set1_epi8(0x0f);
        auto lo = _mm256_and_si256(a, low_mask);
        auto hi = _mm256_and_si256(_mm256_srli_epi16(a, 4), low_mask);
        auto cnt_lo = _mm256_shuffle_epi8(lookup, lo);
        auto cnt_hi = _mm256_shuffle_epi8(lookup, hi);
        return _mm256_add_epi8(cnt_lo, cnt_hi);
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi8(static_cast<int8_t>(0x5B));
        auto pattern2 = _mm256_set1_epi8(static_cast<int8_t>(0x6D));
        auto pattern3 = _mm256_set1_epi8(static_cast<int8_t>(0x8E));
        auto pattern4 = _mm256_set1_epi8(static_cast<int8_t>(0xF0));
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = _mm256_slli_epi16(tmp2, shift);
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(_mm256_add_epi8(popcount(data), tmp2), mask);
        codebits = _mm256_or_si256(codebits, tmp1);
        return codebits;
    }
};

template<>
struct SIMD<__m256i, uint16_t> {
    static __m256i set1(
            uint16_t value) {
        return _mm256_set1_epi16(value);
    }

    static __m256i set_inc(
            uint16_t base = 0,
            uint16_t inc = 1) {
        return _mm256_set_epi16(static_cast<uint16_t>(base + 15 * inc), static_cast<uint16_t>(base + 14 * inc), static_cast<uint16_t>(base + 13 * inc), static_cast<uint16_t>(base + 12 * inc),
                static_cast<uint16_t>(base + 11 * inc), static_cast<uint16_t>(base + 10 * inc), static_cast<uint16_t>(base + 9 * inc), static_cast<uint16_t>(base + 8 * inc),
                static_cast<uint16_t>(base + 7 * inc), static_cast<uint16_t>(base + 6 * inc), static_cast<uint16_t>(base + 5 * inc), static_cast<uint16_t>(base + 4 * inc),
                static_cast<uint16_t>(base + 3 * inc), static_cast<uint16_t>(base + 2 * inc), static_cast<uint16_t>(base + 1 * inc), base);
    }

    static __m256i add(
            __m256i a,
            __m256i b) {
        return _mm256_add_epi16(a, b);
    }

    static __m256i popcount(
            __m256i a) {
        auto mask = _mm256_set1_epi16(0x0101);
        auto shuffle = _mm256_set_epi64x(0xFF0FFF0DFF0BFF09, 0xFF07FF05FF03FF01, 0xFF0FFF0DFF0BFF09, 0xFF07FF05FF03FF01);
        auto popcount8 = SIMD<__m256i, uint8_t>::popcount(a);
        return _mm256_shuffle_epi8(_mm256_mullo_epi16(popcount8, mask), shuffle);
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi16(static_cast<int16_t>(0xAD5B));
        auto pattern2 = _mm256_set1_epi16(static_cast<int16_t>(0x366D));
        auto pattern3 = _mm256_set1_epi16(static_cast<int16_t>(0xC78E));
        auto pattern4 = _mm256_set1_epi16(static_cast<int16_t>(0x07F0));
        auto pattern5 = _mm256_set1_epi16(static_cast<int16_t>(0xF800));
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = _mm256_slli_epi16(tmp2, shift);
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern5)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(_mm256_add_epi8(popcount(data), tmp2), mask);
        codebits = _mm256_or_si256(codebits, tmp1);
        return codebits;
    }
};

template<>
struct SIMD<__m256i, uint32_t> {
    static __m256i set1(
            uint32_t value) {
        return _mm256_set1_epi32(value);
    }

    static __m256i set_inc(
            uint32_t base = 0,
            uint32_t inc = 1) {
        return _mm256_set_epi32(static_cast<uint32_t>(base + 7 * inc), static_cast<uint32_t>(base + 6 * inc), static_cast<uint32_t>(base + 5 * inc), static_cast<uint32_t>(base + 4 * inc),
                static_cast<uint32_t>(base + 3 * inc), static_cast<uint32_t>(base + 2 * inc), static_cast<uint32_t>(base + 1 * inc), base);
    }

    static __m256i add(
            __m256i a,
            __m256i b) {
        return _mm256_add_epi32(a, b);
    }

    static __m256i popcount(
            __m256i a) {
        auto mask = _mm256_set1_epi8(0x01);
        auto shuffle = _mm256_set_epi64x(0xFFFFFF0FFFFFFF0B, 0xFFFFFF07FFFFFF03, 0xFFFFFF0FFFFFFF0B, 0xFFFFFF07FFFFFF03);
        auto popcount8 = SIMD<__m256i, uint8_t>::popcount(a);
        return _mm256_shuffle_epi8(_mm256_mullo_epi32(popcount8, mask), shuffle);
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi32(0x56AAAD5B);
        auto pattern2 = _mm256_set1_epi32(0x9B33366D);
        auto pattern3 = _mm256_set1_epi32(0xE3C3C78E);
        auto pattern4 = _mm256_set1_epi32(0x03FC07F0);
        auto pattern5 = _mm256_set1_epi32(0x03FFF800);
        auto pattern6 = _mm256_set1_epi32(0xFC000000);
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = _mm256_slli_epi32(tmp2, shift);
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern5)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern6)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(_mm256_add_epi8(popcount(data), tmp2), mask);
        codebits = _mm256_or_si256(codebits, tmp1);
        return codebits;
    }
};

template<>
struct SIMD<__m256i, uint64_t> {
    static __m256i set1(
            uint64_t value) {
        return _mm256_set1_epi64x(value);
    }

    static __m256i set_inc(
            uint64_t base = 0,
            uint64_t inc = 1) {
        return _mm256_set_epi64x(static_cast<uint64_t>(base + 3 * inc), static_cast<uint64_t>(base + 2 * inc), static_cast<uint64_t>(base + 1 * inc), base);
    }

    static __m256i add(
            __m256i a,
            __m256i b) {
        return _mm256_add_epi64(a, b);
    }

    static __m256i popcount(
            __m256i a) {
        return _mm256_set_epi64x(__builtin_popcountll(_mm_extract_epi64(a, 3)), __builtin_popcountll(_mm_extract_epi64(a, 2)), __builtin_popcountll(_mm_extract_epi64(a, 1)), __builtin_popcountll(_mm_extract_epi64(a, 0)));
    }
};
#endif

template<>
struct Scalar<uint8_t> {
    static uint8_t popcount(
            uint8_t value) {
        return static_cast<uint8_t>(__builtin_popcount(value));
    }
};

template<>
struct Scalar<uint16_t> {
    static uint16_t popcount(
            uint16_t value) {
        return static_cast<uint16_t>(__builtin_popcount(value));
    }
};

template<>
struct Scalar<uint32_t> {
    static uint32_t popcount(
            uint32_t value) {
        return static_cast<uint32_t>(__builtin_popcountl(value));
    }
};

template<>
struct Scalar<uint64_t> {
    static uint64_t popcount(
            uint64_t value) {
        return static_cast<uint64_t>(__builtin_popcountll(value));
    }
};

struct ExtHamming08 {
    typedef uint64_t accumulator_t;
    typedef uint8_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 8;
    static const constexpr size_t BITCNT_HAMMING = 5;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x5B) & 0x1) << 1;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x6D) & 0x1) << 2;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x8E) & 0x1) << 3;
        hamming |= (Scalar<uint8_t>::popcount(value & 0xF0) & 0x1) << 4;
        hamming |= (Scalar<uint8_t>::popcount(value & 0xFF) + Scalar<uint8_t>::popcount(hamming)) & 0x1;
        return (value << 5) | hamming;
    }
};

struct ExtHamming16 {
    typedef uint64_t accumulator_t;
    typedef uint16_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 16;
    static const constexpr size_t BITCNT_HAMMING = 6;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint16_t>::popcount(value & 0xAD5B) & 0x1) << 1;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x366D) & 0x1) << 2;
        hamming |= (Scalar<uint16_t>::popcount(value & 0xC78E) & 0x1) << 3;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x07F0) & 0x1) << 4;
        hamming |= (Scalar<uint16_t>::popcount(value & 0xF800) & 0x1) << 5;
        hamming |= (Scalar<uint16_t>::popcount(value & 0xFFFF) + Scalar<uint16_t>::popcount(hamming)) & 0x1;
        return (value << 6) | hamming;
    }
};

struct ExtHamming24 {
    typedef uint64_t accumulator_t;
    typedef uint32_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 24;
    static const constexpr size_t BITCNT_HAMMING = 6;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xAAAD5B) & 0x1) << 1;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x33366D) & 0x1) << 2;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xC3C78E) & 0x1) << 3;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xFC07F0) & 0x1) << 4;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xFFF800) & 0x1) << 5;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xFFFFFF) + Scalar<uint32_t>::popcount(hamming)) & 0x1;
        return (value << 6) | hamming;
    }
};

struct ExtHamming32 {
    typedef typename boost::multiprecision::uint128_t accumulator_t;
    typedef uint32_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 32;
    static const constexpr size_t BITCNT_HAMMING = 7;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x56AAAD5B) & 0x1) << 1;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x9B33366D) & 0x1) << 2;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xE3C3C78E) & 0x1) << 3;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x03FC07F0) & 0x1) << 4;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x03FFF800) & 0x1) << 5;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xFC000000) & 0x1) << 6;
        hamming |= (Scalar<uint32_t>::popcount(value & 0xFFFFFFFF) + Scalar<uint32_t>::popcount(hamming)) & 0x1;
        return (value << 7) | hamming;
    }
};

template<size_t BITCNT_DATA>
struct ExtHamming {
    static const constexpr size_t BITCNT_HAMMING = (BITCNT_DATA == 1 ? 3 : (BITCNT_DATA < 5 ? 4 : (BITCNT_DATA < 12 ? 5 : (BITCNT_DATA < 27 ? 6 : (BITCNT_DATA < 58 ? 7 : 8)))));

    typedef typename std::conditional<BITCNT_DATA <= 8, ExtHamming08,
            typename std::conditional<BITCNT_DATA <= 16, ExtHamming16, typename std::conditional<BITCNT_DATA <= 32, ExtHamming32, void>::type>::type>::type hamming_impl_t;
};

template<size_t ACTUAL_BITCNT_DATA>
void countHammingUndetectableErrors() {
    const constexpr size_t BITCNT_CODEWORD = ACTUAL_BITCNT_DATA + ExtHamming<ACTUAL_BITCNT_DATA>::BITCNT_HAMMING;
    const constexpr size_t CNT_COUNTS = BITCNT_CODEWORD + 1ull;

#ifndef NDEBUG
    std::cout << "BITCNT_CODEWORD=" << BITCNT_CODEWORD << '\n';
    std::cout << "CNT_COUNTS=" << CNT_COUNTS << '\n';
#pragma omp parallel
    {
#pragma omp master
        {
            std::cout << "OpenMP using " << omp_get_num_threads() << " threads." << std::endl;
        }
    }
#endif

    typedef typename ExtHamming<ACTUAL_BITCNT_DATA>::hamming_impl_t T;
    typedef typename T::accumulator_t accumulator_t;
    typedef typename T::popcount_t popcount_t;

#if defined (_MSC_VER) and not defined (size_t)
    typedef __int64 size_t;
#endif

    StopWatch sw;
    sw.start();

    accumulator_t * counts = new accumulator_t[CNT_COUNTS]();

#ifndef NDEBUG
    size_t bucket_size_tmp;
    size_t remaining_tmp;
#endif

#ifndef NDEBUG
#pragma omp parallel shared(counts,bucket_size_tmp,remaining_tmp)
#else
#pragma omp parallel shared(counts) num_threads(1)
#endif
    {
        const size_t num_threads = omp_get_num_threads();
        const size_t bucket_size = (0x1ull << ACTUAL_BITCNT_DATA) / num_threads;
        const size_t remaining = (0x1ull << ACTUAL_BITCNT_DATA) % num_threads;

#ifndef NDEBUG
#pragma omp master
        {
            bucket_size_tmp = bucket_size;
            remaining_tmp = remaining;
        }
#endif

#pragma omp for schedule(dynamic,1)
        for (size_t thread = 0; thread < num_threads; ++thread) {
            accumulator_t * counts_local = new accumulator_t[CNT_COUNTS]();
            size_t x = bucket_size * thread;
            const size_t max = x + bucket_size + (thread == (num_threads - 1) ? remaining : 0); // the last thread does the few additional remaining code words

// #ifdef __AVX2__
//             const constexpr size_t values_per_mm256 = sizeof(__m256i) / sizeof(popcount_t);
//             auto mm256 = SIMD<__m256i, popcount_t>::set_inc(x, 1);
//             auto mm256inc = SIMD<__m256i, popcount_t>::set1(values_per_mm256);
//
//             for (; x <= (max - values_per_mm256); x += values_per_mm256) {
//                 auto popcount = SIMD<__m256i, popcount_t>::popcount(mm256);
//                 auto hamming = SIMD<__m256i, popcount_t>::ext_hamming(mm256);
//                 auto hammingPopcount = SIMD<__m256i, popcount_t>::popcount(hamming);
//                 popcount = SIMD<__m256i, popcount_t>::add(popcount, hammingPopcount);
//                 popcount_t * pPopcount = reinterpret_cast<popcount_t*>(&popcount);
//                 for (size_t i = 0; i < values_per_mm256; ++i) {
//                     counts_local[pPopcount[i]]++;
//                 }
//                 mm256 = SIMD<__m256i, popcount_t>::add(mm256, mm256inc);
//             }
// #endif /* __AVX2__ */

#ifdef __SSE4_2__
            const constexpr size_t values_per_mm128 = sizeof(__m128i) / sizeof(popcount_t);
            auto mm128 = SIMD<__m128i, popcount_t>::set_inc(x, 1);
            auto mm128inc = SIMD<__m128i, popcount_t>::set1(values_per_mm128);

            if (max > values_per_mm128) {
                for (; x <= (max - values_per_mm128); x += values_per_mm128) {
                    auto popcount = SIMD<__m128i, popcount_t>::popcount(mm128);
                    auto hamming = SIMD<__m128i, popcount_t>::ext_hamming(mm128);
                    auto hammingPopcount = SIMD<__m128i, popcount_t>::popcount(hamming);
                    popcount = SIMD<__m128i, popcount_t>::add(popcount, hammingPopcount);
                    popcount_t * pPopcount = reinterpret_cast<popcount_t*>(&popcount);
                    for (size_t i = 0; i < values_per_mm128; ++i) {
                        counts_local[pPopcount[i]]++;
                    }
                    mm128 = SIMD<__m128i, popcount_t>::add(mm128, mm128inc);
                }
            }
#endif /* __SSE4_2__ */

            for (; x < max; ++x) {
                counts_local[__builtin_popcountll(T::compute(x))]++;
            }

            // 4) Sum the counts
#pragma omp critical
            for (size_t i = 0; i < CNT_COUNTS; ++i) {
                counts[i] += counts_local[i];
            }
        }
    }
    sw.stop();

    accumulator_t * act_counts = new accumulator_t[CNT_COUNTS]();
    accumulator_t numCodeWords = 0x1ull << ACTUAL_BITCNT_DATA;

    accumulator_t maxCounts = 0;

    // the transitions apply to all code words
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        act_counts[i] = counts[i] * numCodeWords;
        if (counts[i] > maxCounts) {
            maxCounts = counts[i];
        }
    }
    size_t maxWidth0 = 0;
    do {
        maxCounts /= 10;
        ++maxWidth0;
    } while (maxCounts);

    // the 1-bit sphere transitions
    for (size_t i = 1; i < CNT_COUNTS; i += 2) {
        act_counts[i] = (act_counts[i - 1] * (BITCNT_CODEWORD - (i - 1))) + ((i + 1) < CNT_COUNTS ? (act_counts[i + 1] * (i + 1)) : 0);
    }

    accumulator_t max = 0;
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        if (act_counts[i] > max) {
            max = act_counts[i];
        }
    }
    size_t maxWidth1 = 0;
    do {
        max /= 10;
        ++maxWidth1;
    } while (max);
    size_t maxWidth2 = 0;
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        size_t maxWidth2tmp = 0;
        accumulator_t maxTransitions2 = numCodeWords * binomial_coefficient<accumulator_t>(BITCNT_CODEWORD, i);
        do {
            maxTransitions2 /= 10;
            ++maxWidth2tmp;
        } while (maxTransitions2 > 0);
        if (maxWidth2tmp > maxWidth2) {
            maxWidth2 = maxWidth2tmp;
        }
    }
    accumulator_t numTotal = 0;
    accumulator_t numTransitions = 0;
#ifndef NDEBUG
    std::cout << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : bucketsize(" << bucket_size_tmp << ") remaining(" << remaining_tmp << ")" << std::endl;
#endif
    std::cout << std::dec << "#Distances:\n" << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        accumulator_t maxTransitions = numCodeWords * binomial_coefficient<accumulator_t>(BITCNT_CODEWORD, i);
        numTotal += counts[i];
        numTransitions += act_counts[i];
        double probability = double(act_counts[i]) / double(maxTransitions);
        std::cout << std::right << std::setw(4) << i << ',' << std::setw(maxWidth0) << counts[i] << ',' << std::setw(maxWidth1) << act_counts[i] << ',' << std::setw(maxWidth2) << maxTransitions << ','
                << probability << '\n';
    }
    if (numTotal != numCodeWords) {
        std::cerr << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : numTotal (" << numTotal << " != numCodeWords (" << numCodeWords << ')' << std::endl;
    }

    std::cout << "#(" << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : Computation took " << sw << "ns for " << numCodeWords << " code words and " << numTransitions << " transitions."
            << std::endl;
}

int main() {
    countHammingUndetectableErrors<1>();
    countHammingUndetectableErrors<2>();
    countHammingUndetectableErrors<3>();
    countHammingUndetectableErrors<4>();
    countHammingUndetectableErrors<5>();
    countHammingUndetectableErrors<6>();
    countHammingUndetectableErrors<7>();
    countHammingUndetectableErrors<8>();
    countHammingUndetectableErrors<9>();
    countHammingUndetectableErrors<10>();
    countHammingUndetectableErrors<11>();
    countHammingUndetectableErrors<12>();
    countHammingUndetectableErrors<13>();
    countHammingUndetectableErrors<14>();
    countHammingUndetectableErrors<15>();
    countHammingUndetectableErrors<16>();
    countHammingUndetectableErrors<17>();
    countHammingUndetectableErrors<18>();
    countHammingUndetectableErrors<19>();
    countHammingUndetectableErrors<20>();
    countHammingUndetectableErrors<21>();
    countHammingUndetectableErrors<22>();
    countHammingUndetectableErrors<23>();
    countHammingUndetectableErrors<24>();
    countHammingUndetectableErrors<25>();
    countHammingUndetectableErrors<26>();
    countHammingUndetectableErrors<27>();
    countHammingUndetectableErrors<28>();
    countHammingUndetectableErrors<29>();
    countHammingUndetectableErrors<30>();
    countHammingUndetectableErrors<31>();
    countHammingUndetectableErrors<32>();

    return 0;
}
