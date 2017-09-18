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
#include <array>
#include <utility>
#include <vector>
#include <type_traits>
#include <omp.h>
#include <immintrin.h>
#include <boost/multiprecision/cpp_int.hpp>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt64
#endif

#if defined(__AVX512F__) and defined(__AVX512ER__) and defined(__AVX512CD__) and defined(__AVX512PF__)
#define __MARCH_KNL__
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

template<typename T>
struct Scalar;

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
    typedef uint32_t accumulator_t;
    typedef uint8_t data_t;

    static const constexpr data_t pattern1 = static_cast<data_t>(0x5B);
    static const constexpr data_t pattern2 = static_cast<data_t>(0x6D);
    static const constexpr data_t pattern3 = static_cast<data_t>(0x8E);
    static const constexpr data_t pattern4 = static_cast<data_t>(0xF0);

    static inline data_t compute(
            const data_t value) {
        data_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
        return hamming;
    }

    static inline data_t compute_ext(
            const data_t value) {
        data_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
        return hamming;
    }
};

struct ExtHamming16 {
    typedef uint64_t accumulator_t;
    typedef uint16_t data_t;

    static const constexpr data_t pattern1 = static_cast<data_t>(0xAD5B);
    static const constexpr data_t pattern2 = static_cast<data_t>(0x366D);
    static const constexpr data_t pattern3 = static_cast<data_t>(0xC78E);
    static const constexpr data_t pattern4 = static_cast<data_t>(0x07F0);
    static const constexpr data_t pattern5 = static_cast<data_t>(0xF800);

    static inline data_t compute(
            const data_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 4;
        return hamming;
    }

    static inline data_t compute_ext(
            const data_t value) {
        data_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 5;
        hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
        return hamming;
    }
};

struct ExtHamming32 {
    typedef typename boost::multiprecision::uint128_t accumulator_t;
    typedef uint32_t data_t;

    static const constexpr data_t pattern1 = 0x56AAAD5B;
    static const constexpr data_t pattern2 = 0x9B33366D;
    static const constexpr data_t pattern3 = 0xE3C3C78E;
    static const constexpr data_t pattern4 = 0x03FC07F0;
    static const constexpr data_t pattern5 = 0x03FFF800;
    static const constexpr data_t pattern6 = 0xFC000000;

    static inline data_t compute(
            const data_t value) {
        data_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 5;
        return hamming;
    }

    static inline size_t compute_ext(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 5;
        hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 6;
        hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
        return hamming;
    }
};

struct ExtHamming64 {
    typedef typename boost::multiprecision::uint128_t accumulator_t;
    typedef uint64_t data_t;

    static const constexpr data_t pattern1 = 0xAB55555556AAAD5Bull;
    static const constexpr data_t pattern2 = 0xCD9999999B33366Dull;
    static const constexpr data_t pattern3 = 0x78F1E1E1E3C3C78Eull;
    static const constexpr data_t pattern4 = 0x01FE01FE03FC07F0ull;
    static const constexpr data_t pattern5 = 0x01FFFE0003FFF800ull;
    static const constexpr data_t pattern6 = 0x01FFFFFFFC000000ull;
    static const constexpr data_t pattern7 = 0xFE00000000000000ull;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 5;
        hamming |= (Scalar<data_t>::popcount(value & pattern7) & 0x1) << 6;
        return hamming;
    }

    static inline size_t compute_ext(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
        hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
        hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
        hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
        hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 5;
        hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 6;
        hamming |= (Scalar<data_t>::popcount(value & pattern7) & 0x1) << 7;
        hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
        return hamming;
    }
};

template<size_t BITCNT_DATA>
struct ExtHamming {
    static const constexpr size_t BITCNT_HAMMING = (
            BITCNT_DATA == 1 ? 3 : (BITCNT_DATA <= 4 ? 4 : (BITCNT_DATA <= 11 ? 5 : (BITCNT_DATA <= 26 ? 6 : (BITCNT_DATA <= 57 ? 7 : BITCNT_DATA <= 64 ? 8 : 9)))));

    typedef typename std::conditional<BITCNT_DATA <= 8, ExtHamming08,
            typename std::conditional<BITCNT_DATA <= 16, ExtHamming16,
                    typename std::conditional<BITCNT_DATA <= 32, ExtHamming32, typename std::conditional<BITCNT_DATA <= 64, ExtHamming64, void>::type>::type>::type>::type hamming_impl_t;
    typedef typename hamming_impl_t::data_t data_t;
};

template<typename V, typename T>
struct SIMD;

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

    static __m128i hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi8(ExtHamming08::pattern1);
        auto pattern2 = _mm_set1_epi8(ExtHamming08::pattern2);
        auto pattern3 = _mm_set1_epi8(ExtHamming08::pattern3);
        auto pattern4 = _mm_set1_epi8(ExtHamming08::pattern4);
        auto mask = _mm_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern3)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern4)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        return codebits;
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi8(ExtHamming08::pattern1);
        auto pattern2 = _mm_set1_epi8(ExtHamming08::pattern2);
        auto pattern3 = _mm_set1_epi8(ExtHamming08::pattern3);
        auto pattern4 = _mm_set1_epi8(ExtHamming08::pattern4);
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

    static __m128i hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi16(ExtHamming16::pattern1);
        auto pattern2 = _mm_set1_epi16(ExtHamming16::pattern2);
        auto pattern3 = _mm_set1_epi16(ExtHamming16::pattern3);
        auto pattern4 = _mm_set1_epi16(ExtHamming16::pattern4);
        auto pattern5 = _mm_set1_epi16(ExtHamming16::pattern5);
        auto mask = _mm_set1_epi16(static_cast<int16_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern3)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern4)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern5)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi16(tmp1, ++shift));
        return codebits;
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi16(ExtHamming16::pattern1);
        auto pattern2 = _mm_set1_epi16(ExtHamming16::pattern2);
        auto pattern3 = _mm_set1_epi16(ExtHamming16::pattern3);
        auto pattern4 = _mm_set1_epi16(ExtHamming16::pattern4);
        auto pattern5 = _mm_set1_epi16(ExtHamming16::pattern5);
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

    static __m128i hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi32(ExtHamming32::pattern1);
        auto pattern2 = _mm_set1_epi32(ExtHamming32::pattern2);
        auto pattern3 = _mm_set1_epi32(ExtHamming32::pattern3);
        auto pattern4 = _mm_set1_epi32(ExtHamming32::pattern4);
        auto pattern5 = _mm_set1_epi32(ExtHamming32::pattern5);
        auto pattern6 = _mm_set1_epi32(ExtHamming32::pattern6);
        auto mask = _mm_set1_epi32(0x1);
        uint8_t shift = 1;
        __m128i tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        __m128i codebits = tmp2;
        __m128i tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, shift));
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
        return codebits;
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi32(ExtHamming32::pattern1);
        auto pattern2 = _mm_set1_epi32(ExtHamming32::pattern2);
        auto pattern3 = _mm_set1_epi32(ExtHamming32::pattern3);
        auto pattern4 = _mm_set1_epi32(ExtHamming32::pattern4);
        auto pattern5 = _mm_set1_epi32(ExtHamming32::pattern5);
        auto pattern6 = _mm_set1_epi32(ExtHamming32::pattern6);
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
        auto popcnt32 = SIMD<__m128i, uint32_t>::popcount(a);
        return _mm_and_si128(_mm_set1_epi64x(0xFFull), _mm_add_epi32(popcnt32, _mm_srli_epi64(popcnt32, 32)));
    }

    static __m128i hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi64x(ExtHamming64::pattern1);
        auto pattern2 = _mm_set1_epi64x(ExtHamming64::pattern2);
        auto pattern3 = _mm_set1_epi64x(ExtHamming64::pattern3);
        auto pattern4 = _mm_set1_epi64x(ExtHamming64::pattern4);
        auto pattern5 = _mm_set1_epi64x(ExtHamming64::pattern5);
        auto pattern6 = _mm_set1_epi64x(ExtHamming64::pattern6);
        auto pattern7 = _mm_set1_epi64x(ExtHamming64::pattern7);
        auto mask = _mm_set1_epi32(0x1);
        uint8_t shift = 1;
        __m128i tmp2 = _mm_and_si128(popcount(_mm_and_si128(data, pattern1)), mask);
        __m128i codebits = tmp2;
        __m128i tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern2)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, shift));
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
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern7)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        return codebits;
    }

    static __m128i ext_hamming(
            __m128i data) {
        auto pattern1 = _mm_set1_epi64x(ExtHamming64::pattern1);
        auto pattern2 = _mm_set1_epi64x(ExtHamming64::pattern2);
        auto pattern3 = _mm_set1_epi64x(ExtHamming64::pattern3);
        auto pattern4 = _mm_set1_epi64x(ExtHamming64::pattern4);
        auto pattern5 = _mm_set1_epi64x(ExtHamming64::pattern5);
        auto pattern6 = _mm_set1_epi64x(ExtHamming64::pattern6);
        auto pattern7 = _mm_set1_epi64x(ExtHamming64::pattern7);
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
        tmp1 = _mm_and_si128(popcount(_mm_and_si128(data, pattern7)), mask);
        tmp2 = _mm_xor_si128(tmp2, tmp1);
        codebits = _mm_or_si128(codebits, _mm_slli_epi32(tmp1, ++shift));
        codebits = _mm_or_si128(codebits, _mm_and_si128(_mm_add_epi32(popcount(data), tmp2), mask));
        return codebits;
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

    static __m256i hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi8(ExtHamming08::pattern1);
        auto pattern2 = _mm256_set1_epi8(ExtHamming08::pattern2);
        auto pattern3 = _mm256_set1_epi8(ExtHamming08::pattern3);
        auto pattern4 = _mm256_set1_epi8(ExtHamming08::pattern4);
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        return codebits;
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi8(ExtHamming08::pattern1);
        auto pattern2 = _mm256_set1_epi8(ExtHamming08::pattern2);
        auto pattern3 = _mm256_set1_epi8(ExtHamming08::pattern3);
        auto pattern4 = _mm256_set1_epi8(ExtHamming08::pattern4);
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

    static __m256i hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi16(ExtHamming16::pattern1);
        auto pattern2 = _mm256_set1_epi16(ExtHamming16::pattern2);
        auto pattern3 = _mm256_set1_epi16(ExtHamming16::pattern3);
        auto pattern4 = _mm256_set1_epi16(ExtHamming16::pattern4);
        auto pattern5 = _mm256_set1_epi16(ExtHamming16::pattern5);
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern5)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
        return codebits;
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi16(ExtHamming16::pattern1);
        auto pattern2 = _mm256_set1_epi16(ExtHamming16::pattern2);
        auto pattern3 = _mm256_set1_epi16(ExtHamming16::pattern3);
        auto pattern4 = _mm256_set1_epi16(ExtHamming16::pattern4);
        auto pattern5 = _mm256_set1_epi16(ExtHamming16::pattern5);
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

    static __m256i hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi32(ExtHamming32::pattern1);
        auto pattern2 = _mm256_set1_epi32(ExtHamming32::pattern2);
        auto pattern3 = _mm256_set1_epi32(ExtHamming32::pattern3);
        auto pattern4 = _mm256_set1_epi32(ExtHamming32::pattern4);
        auto pattern5 = _mm256_set1_epi32(ExtHamming32::pattern5);
        auto pattern6 = _mm256_set1_epi32(ExtHamming32::pattern6);
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, shift));
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
        return codebits;
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi32(ExtHamming32::pattern1);
        auto pattern2 = _mm256_set1_epi32(ExtHamming32::pattern2);
        auto pattern3 = _mm256_set1_epi32(ExtHamming32::pattern3);
        auto pattern4 = _mm256_set1_epi32(ExtHamming32::pattern4);
        auto pattern5 = _mm256_set1_epi32(ExtHamming32::pattern5);
        auto pattern6 = _mm256_set1_epi32(ExtHamming32::pattern6);
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
        auto popcnt32 = SIMD<__m256i, uint32_t>::popcount(a);
        return _mm256_and_si256(_mm256_set1_epi64x(0xFFull), _mm256_add_epi32(popcnt32, _mm256_srli_epi64(popcnt32, 32)));
    }

    static __m256i hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi64x(ExtHamming64::pattern1);
        auto pattern2 = _mm256_set1_epi64x(ExtHamming64::pattern2);
        auto pattern3 = _mm256_set1_epi64x(ExtHamming64::pattern3);
        auto pattern4 = _mm256_set1_epi64x(ExtHamming64::pattern4);
        auto pattern5 = _mm256_set1_epi64x(ExtHamming64::pattern5);
        auto pattern6 = _mm256_set1_epi64x(ExtHamming64::pattern6);
        auto pattern7 = _mm256_set1_epi64x(ExtHamming64::pattern7);
        auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
        uint8_t shift = 1;
        auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
        auto codebits = tmp2;
        auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, shift));
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
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern7)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        return codebits;
    }

    static __m256i ext_hamming(
            __m256i data) {
        auto pattern1 = _mm256_set1_epi64x(ExtHamming64::pattern1);
        auto pattern2 = _mm256_set1_epi64x(ExtHamming64::pattern2);
        auto pattern3 = _mm256_set1_epi64x(ExtHamming64::pattern3);
        auto pattern4 = _mm256_set1_epi64x(ExtHamming64::pattern4);
        auto pattern5 = _mm256_set1_epi64x(ExtHamming64::pattern5);
        auto pattern6 = _mm256_set1_epi64x(ExtHamming64::pattern6);
        auto pattern7 = _mm256_set1_epi64x(ExtHamming64::pattern7);
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
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern7)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi32(tmp1, ++shift));
        tmp1 = _mm256_and_si256(_mm256_add_epi8(popcount(data), tmp2), mask);
        codebits = _mm256_or_si256(codebits, tmp1);
        return codebits;
    }
};
#endif

/*
 #ifdef __MARCH_KNL__

 template<>
 struct SIMD<__m512i, uint8_t> {
 static __m512i set1(
 uint8_t value) {
 return _mm512_set1_epi8(value);
 }

 static __m512i set_inc(
 uint8_t base = 0,
 uint8_t inc = 1) {
 return _mm512_set_epi8(static_cast<uint8_t>(base + 63 * inc), static_cast<uint8_t>(base + 62 * inc), static_cast<uint8_t>(base + 61 * inc), static_cast<uint8_t>(base + 60 * inc),
 static_cast<uint8_t>(base + 59 * inc), static_cast<uint8_t>(base + 58 * inc), static_cast<uint8_t>(base + 57 * inc), static_cast<uint8_t>(base + 56 * inc),
 static_cast<uint8_t>(base + 55 * inc), static_cast<uint8_t>(base + 54 * inc), static_cast<uint8_t>(base + 53 * inc), static_cast<uint8_t>(base + 52 * inc),
 static_cast<uint8_t>(base + 51 * inc), static_cast<uint8_t>(base + 50 * inc), static_cast<uint8_t>(base + 49 * inc), static_cast<uint8_t>(base + 48 * inc),
 static_cast<uint8_t>(base + 47 * inc), static_cast<uint8_t>(base + 46 * inc), static_cast<uint8_t>(base + 45 * inc), static_cast<uint8_t>(base + 44 * inc),
 static_cast<uint8_t>(base + 43 * inc), static_cast<uint8_t>(base + 42 * inc), static_cast<uint8_t>(base + 41 * inc), static_cast<uint8_t>(base + 40 * inc),
 static_cast<uint8_t>(base + 39 * inc), static_cast<uint8_t>(base + 38 * inc), static_cast<uint8_t>(base + 37 * inc), static_cast<uint8_t>(base + 36 * inc),
 static_cast<uint8_t>(base + 35 * inc), static_cast<uint8_t>(base + 34 * inc), static_cast<uint8_t>(base + 33 * inc), static_cast<uint8_t>(base + 32 * inc),
 static_cast<uint8_t>(base + 31 * inc), static_cast<uint8_t>(base + 30 * inc), static_cast<uint8_t>(base + 29 * inc), static_cast<uint8_t>(base + 28 * inc),
 static_cast<uint8_t>(base + 27 * inc), static_cast<uint8_t>(base + 26 * inc), static_cast<uint8_t>(base + 25 * inc), static_cast<uint8_t>(base + 24 * inc),
 static_cast<uint8_t>(base + 23 * inc), static_cast<uint8_t>(base + 22 * inc), static_cast<uint8_t>(base + 21 * inc), static_cast<uint8_t>(base + 20 * inc),
 static_cast<uint8_t>(base + 19 * inc), static_cast<uint8_t>(base + 18 * inc), static_cast<uint8_t>(base + 17 * inc), static_cast<uint8_t>(base + 16 * inc),
 static_cast<uint8_t>(base + 15 * inc), static_cast<uint8_t>(base + 14 * inc), static_cast<uint8_t>(base + 13 * inc), static_cast<uint8_t>(base + 12 * inc),
 static_cast<uint8_t>(base + 11 * inc), static_cast<uint8_t>(base + 10 * inc), static_cast<uint8_t>(base + 9 * inc), static_cast<uint8_t>(base + 8 * inc),
 static_cast<uint8_t>(base + 7 * inc), static_cast<uint8_t>(base + 6 * inc), static_cast<uint8_t>(base + 5 * inc), static_cast<uint8_t>(base + 4 * inc),
 static_cast<uint8_t>(base + 3 * inc), static_cast<uint8_t>(base + 2 * inc), static_cast<uint8_t>(base + 1 * inc), base);
 }

 static __m512i add(
 __m512i a,
 __m512i b) {
 return _mm512_add_epi8(a, b);
 }

 static __m512i popcount(
 __m512i a) {
 auto lookup = _mm512_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
 auto low_mask = _mm512_set1_epi8(0x0f);
 auto lo = _mm512_and_si512(a, low_mask);
 auto hi = _mm512_and_si512(_mm256_srli_epi16(a, 4), low_mask);
 auto cnt_lo = _mm512_shuffle_epi8(lookup, lo);
 auto cnt_hi = _mm512_shuffle_epi8(lookup, hi);
 return _mm256_add_epi8(cnt_lo, cnt_hi);
 }

 static __m256i hamming(
 __m256i data) {
 auto pattern1 = _mm256_set1_epi8(ExtHamming08::pattern1);
 auto pattern2 = _mm256_set1_epi8(ExtHamming08::pattern2);
 auto pattern3 = _mm256_set1_epi8(ExtHamming08::pattern3);
 auto pattern4 = _mm256_set1_epi8(ExtHamming08::pattern4);
 auto mask = _mm256_set1_epi8(static_cast<int8_t>(0x1));
 uint8_t shift = 1;
 auto tmp2 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern1)), mask);
 auto codebits = tmp2;
 auto tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern2)), mask);
 tmp2 = _mm256_xor_si256(tmp2, tmp1);
 codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, shift));
 tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern3)), mask);
 tmp2 = _mm256_xor_si256(tmp2, tmp1);
 codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
 tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern4)), mask);
 tmp2 = _mm256_xor_si256(tmp2, tmp1);
 codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
 return codebits;
 }

 static __m256i ext_hamming(
 __m256i data) {
 auto pattern1 = _mm256_set1_epi8(ExtHamming08::pattern1);
 auto pattern2 = _mm256_set1_epi8(ExtHamming08::pattern2);
 auto pattern3 = _mm256_set1_epi8(ExtHamming08::pattern3);
 auto pattern4 = _mm256_set1_epi8(ExtHamming08::pattern4);
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
 #endif
 */

struct statistics {
    const size_t numDataBits;
    const size_t numHammingBits;
    const size_t numCodeBits;
    const boost::multiprecision::uint128_t numCodeWords;
    const size_t numCounts;
    std::unique_ptr<boost::multiprecision::uint128_t[]> counts;
    std::unique_ptr<boost::multiprecision::uint128_t[]> ext_counts;
    std::unique_ptr<boost::multiprecision::uint128_t[]> act_counts;

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = 0;
            this->ext_counts[i] = 0;
            this->act_counts[i] = 0;
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            boost::multiprecision::uint128_t * counts,
            boost::multiprecision::uint128_t * ext_counts,
            boost::multiprecision::uint128_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
            this->ext_counts[i] = ext_counts[i];
            this->act_counts[i] = act_counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint64_t * counts,
            uint64_t * ext_counts,
            uint64_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
            this->ext_counts[i] = ext_counts[i];
            this->act_counts[i] = act_counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint32_t * counts,
            uint32_t * ext_counts,
            uint32_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
            this->ext_counts[i] = ext_counts[i];
            this->act_counts[i] = act_counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint16_t * counts,
            uint16_t * ext_counts,
            uint16_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
            this->ext_counts[i] = ext_counts[i];
            this->act_counts[i] = act_counts[i];
        }
    }

    statistics(
            size_t numDataBits,
            size_t numHammingBits,
            size_t numCodeBits,
            boost::multiprecision::uint128_t numCodeWords,
            size_t numCounts,
            uint8_t * counts,
            uint8_t * ext_counts,
            uint8_t * act_counts)
            : numDataBits(numDataBits),
              numHammingBits(numHammingBits),
              numCodeBits(numCodeBits),
              numCodeWords(numCodeWords),
              numCounts(numCounts),
              counts(new boost::multiprecision::uint128_t[numCounts]),
              ext_counts(new boost::multiprecision::uint128_t[numCounts]),
              act_counts(new boost::multiprecision::uint128_t[numCounts]) {
        for (size_t i = 0; i < numCounts; ++i) {
            this->counts[i] = counts[i];
            this->ext_counts[i] = ext_counts[i];
            this->act_counts[i] = act_counts[i];
        }
    }

    statistics(
            statistics && other)
            : numDataBits(other.numDataBits),
              numHammingBits(other.numHammingBits),
              numCodeBits(other.numCodeBits),
              numCodeWords(other.numCodeWords),
              numCounts(other.numCounts),
              counts(std::move(other.counts)),
              ext_counts(std::move(other.ext_counts)),
              act_counts(std::move(other.act_counts)) {
    }

    statistics(
            statistics & other) = delete;

    statistics & operator=(
            statistics && other) {
        new (this) statistics(std::forward<statistics>(other));
        return *this;
    }

    statistics & operator=(
            statistics & other) = delete;
};

template<size_t ACTUAL_BITCNT_DATA>
statistics countHammingUndetectableErrors(
        statistics & former_stats) {
    typedef typename ExtHamming<ACTUAL_BITCNT_DATA>::hamming_impl_t T;
    typedef typename T::accumulator_t accumulator_t;
    typedef typename T::data_t data_t;

    const constexpr size_t BITCNT_HAMMINGBITS = ExtHamming<ACTUAL_BITCNT_DATA>::BITCNT_HAMMING;
    const constexpr size_t BITCNT_CODEWORD = ACTUAL_BITCNT_DATA + BITCNT_HAMMINGBITS;
    const constexpr accumulator_t CNT_CODEWORDS = 1ull << ACTUAL_BITCNT_DATA;
    const constexpr size_t CNT_COUNTS = BITCNT_CODEWORD + 1ull;

#if defined (_MSC_VER) and not defined (size_t)
    typedef __int64 size_t;
#endif

    StopWatch sw;
    sw.start();

    statistics stats(ACTUAL_BITCNT_DATA, BITCNT_HAMMINGBITS, BITCNT_CODEWORD, CNT_CODEWORDS, CNT_COUNTS);
    // we reuse the former stats to half the number of code words we actually need to count
#ifndef NDEBUG
    std::cout << "# former stats given (" << former_stats.numCounts << " counts, " << former_stats.numCodeWords << " codewords):\n";
#endif
    for (size_t i = 0; i < former_stats.numCounts; ++i) {
        stats.counts[i] = former_stats.counts[i];
        stats.ext_counts[i] = former_stats.ext_counts[i];
        stats.act_counts[i] = former_stats.act_counts[i];
#ifndef NDEBUG
        std::cout << "#\t" << stats.counts[i] << "," << stats.ext_counts[i] << "," << stats.act_counts[i] << "\n";
#endif
    }
    std::cout << std::flush;

#ifndef NDEBUG
    std::cout << "BITCNT_HAMMINGBITS=" << BITCNT_HAMMINGBITS << "\n";
    std::cout << "BITCNT_CODEWORD=" << BITCNT_CODEWORD << "\n";
    std::cout << "CNT_CODEWORDS=" << CNT_CODEWORDS << "\n";
    std::cout << "CNT_COUNTS=" << CNT_COUNTS << "\n";
#pragma omp parallel
    {
#pragma omp master
        {
            std::cout << "OpenMP using " << omp_get_num_threads() << " threads." << std::endl;
        }
    }
    size_t bucket_size_tmp;
    size_t remaining_tmp;
#pragma omp parallel shared(stats,former_stats,bucket_size_tmp,remaining_tmp)
#else
#pragma omp parallel shared(stats,former_stats)
#endif
    {
        const size_t num_threads = omp_get_num_threads();
        const size_t numCodeWordsToEnumerate = CNT_CODEWORDS - former_stats.numCodeWords;
        const size_t bucket_size = numCodeWordsToEnumerate / num_threads;
        const size_t remaining = numCodeWordsToEnumerate % num_threads;

#ifndef NDEBUG
#pragma omp master
        {
            bucket_size_tmp = bucket_size;
            remaining_tmp = remaining;
        }
#endif

#pragma omp for schedule(dynamic,1)
        for (size_t thread = 0; thread < num_threads; ++thread) {
            std::array<accumulator_t, CNT_COUNTS> counts_local {0};
            size_t x = former_stats.numCodeWords + bucket_size * thread;
            const size_t max = x + bucket_size + (thread == (num_threads - 1) ? remaining : 0); // the last thread does the few additional remaining code words

#ifdef __AVX2__
            const constexpr size_t values_per_mm256 = sizeof(__m256i) / sizeof(data_t);
            auto mm256 = SIMD<__m256i, data_t>::set_inc(x, 1);
            auto mm256inc = SIMD<__m256i, data_t>::set1(values_per_mm256);

            if (max >= values_per_mm256) {
                for (; x <= (max - values_per_mm256); x += values_per_mm256) {
                    auto popcount = SIMD<__m256i, data_t>::popcount(mm256);
                    auto hamming = SIMD<__m256i, data_t>::hamming(mm256);
                    auto hammingPopcount = SIMD<__m256i, data_t>::popcount(hamming);
                    popcount = SIMD<__m256i, data_t>::add(popcount, hammingPopcount);
                    auto * pPopcount = reinterpret_cast<data_t*>(&popcount);
                    for (size_t i = 0; i < values_per_mm256; ++i) {
                        counts_local[pPopcount[i]]++;
                    }
                    mm256 = SIMD<__m256i, data_t>::add(mm256, mm256inc);
                }
            }
#endif /* __AVX2__ */

#ifdef __SSE4_2__
            const constexpr size_t values_per_mm128 = sizeof(__m128i) / sizeof(data_t);
            auto mm128 = SIMD<__m128i, data_t>::set_inc(x, 1);
            auto mm128inc = SIMD<__m128i, data_t>::set1(values_per_mm128);

            if (max >= values_per_mm128) {
                for (; x <= (max - values_per_mm128); x += values_per_mm128) {
                    auto popcount = SIMD<__m128i, data_t>::popcount(mm128);
                    auto hamming = SIMD<__m128i, data_t>::hamming(mm128);
                    auto hammingPopcount = SIMD<__m128i, data_t>::popcount(hamming);
                    popcount = SIMD<__m128i, data_t>::add(popcount, hammingPopcount);
                    auto * pPopcount = reinterpret_cast<data_t*>(&popcount);
                    for (size_t i = 0; i < values_per_mm128; ++i) {
                        counts_local[pPopcount[i]]++;
                    }
                    mm128 = SIMD<__m128i, data_t>::add(mm128, mm128inc);
                }
            }
#endif /* __SSE4_2__ */

            for (; x < max; ++x) {
                counts_local[Scalar<data_t>::popcount(x) + Scalar<data_t>::popcount(T::compute(x))]++;
            }

            // 4) Sum the counts
#pragma omp critical
            for (size_t i = 0; i < CNT_COUNTS; ++i) {
                stats.counts[i] += counts_local[i];
            }
        }
    }
    sw.stop();

    accumulator_t maxCounts = 0;
    // extend the basic counts to extended Hamming
    stats.ext_counts[0] = stats.counts[0];
    for (size_t i = 1; i < CNT_COUNTS; ++i) {
        if (stats.counts[i] > maxCounts) {
            maxCounts = static_cast<accumulator_t>(stats.counts[i]);
        }
        if (i & 0x1) {
            stats.ext_counts[i] = 0;
        } else {
            stats.ext_counts[i] = stats.counts[i] + stats.counts[i - 1];
        }
    }
    size_t maxWidth1 = 0;
    do {
        maxCounts /= 10;
        ++maxWidth1;
    } while (maxCounts);

    accumulator_t maxCountsExt = 0;
    // the transitions apply to all code words
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        stats.act_counts[i] = stats.ext_counts[i] * CNT_CODEWORDS;
        if (stats.ext_counts[i] > maxCountsExt) {
            maxCountsExt = static_cast<accumulator_t>(stats.ext_counts[i]);
        }
    }
    size_t maxWidth2 = 0;
    do {
        maxCountsExt /= 10;
        ++maxWidth2;
    } while (maxCountsExt);
    // the 1-bit sphere transitions
    for (size_t i = 1; i < CNT_COUNTS; i += 2) {
        stats.act_counts[i] = (stats.act_counts[i - 1] * (BITCNT_CODEWORD - (i - 1))) + ((i + 1) < CNT_COUNTS ? (stats.act_counts[i + 1] * (i + 1)) : 0);
    }

    accumulator_t max = 0;
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        if (stats.act_counts[i] > max) {
            max = static_cast<accumulator_t>(stats.act_counts[i]);
        }
    }
    size_t maxWidth3 = 0;
    do {
        max /= 10;
        ++maxWidth3;
    } while (max);
    size_t maxWidth4 = 0;
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        size_t maxWidth2tmp = 0;
        accumulator_t maxTransitions2 = CNT_CODEWORDS * binomial_coefficient<accumulator_t>(BITCNT_CODEWORD, i);
        do {
            maxTransitions2 /= 10;
            ++maxWidth2tmp;
        } while (maxTransitions2 > 0);
        if (maxWidth2tmp > maxWidth4) {
            maxWidth4 = maxWidth2tmp;
        }
    }
    accumulator_t numTotal = 0;
    accumulator_t numTransitions = 0;
#ifndef NDEBUG
    std::cout << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : bucketsize(" << bucket_size_tmp << ") remaining(" << remaining_tmp << ")" << std::endl;
#endif
    std::cout << "# The reported colunms are:\n";
    std::cout << "#   1) bit width (k)\n" << "#   2) (shortened) detecting-only Hamming count\n" << "#   3) (shortened) detecting-only Extended Hamming count\n"
            << "#   4) (shortened) correcting Extended Hamming count\n" << "#   5) maximum\n#   6) probability (col 4 / (col 5 * n over k))\n";
    std::cout << std::dec << "#Distances:\n" << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        accumulator_t maxTransitions = CNT_CODEWORDS * binomial_coefficient<accumulator_t>(BITCNT_CODEWORD, i);
        numTotal += static_cast<accumulator_t>(stats.counts[i]);
        numTransitions += static_cast<accumulator_t>(stats.act_counts[i]);
        double probability = double(stats.act_counts[i]) / double(maxTransitions);
        std::cout << std::right << std::setw(4) << i << ',' << std::setw(maxWidth1) << stats.counts[i] << ',' << std::setw(maxWidth2) << stats.ext_counts[i] << ',' << std::setw(maxWidth3)
                << stats.act_counts[i] << ',' << std::setw(maxWidth4) << maxTransitions << ',' << probability << '\n';
    }
    if (numTotal != CNT_CODEWORDS) {
        std::cerr << '(' << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : numTotal (" << numTotal << " != numCodeWords (" << CNT_CODEWORDS << ')' << std::endl;
    }

    std::cout << "# (" << BITCNT_CODEWORD << '/' << ACTUAL_BITCNT_DATA << ") : Computation took " << sw << "ns for " << CNT_CODEWORDS << " code words and " << numTransitions << " transitions.\n\n"
            << std::endl;

    return stats;
}

int main() {
    std::vector<statistics> all_stats;
    all_stats.reserve(64);
    all_stats.push_back(statistics(0, 0, 0, 0, 1));
    all_stats.emplace_back(countHammingUndetectableErrors<1>(all_stats[0])); // all stats are zero-initialized
    all_stats.emplace_back(countHammingUndetectableErrors<2>(all_stats[1])); // now stats[0] is all set-up
    all_stats.emplace_back(countHammingUndetectableErrors<3>(all_stats[2]));
    all_stats.emplace_back(countHammingUndetectableErrors<4>(all_stats[3]));
    all_stats.emplace_back(countHammingUndetectableErrors<5>(all_stats[4]));
    all_stats.emplace_back(countHammingUndetectableErrors<6>(all_stats[5]));
    all_stats.emplace_back(countHammingUndetectableErrors<7>(all_stats[6]));
    all_stats.emplace_back(countHammingUndetectableErrors<8>(all_stats[7]));
    all_stats.emplace_back(countHammingUndetectableErrors<9>(all_stats[8]));
    all_stats.emplace_back(countHammingUndetectableErrors<10>(all_stats[9]));
    all_stats.emplace_back(countHammingUndetectableErrors<11>(all_stats[10]));
    all_stats.emplace_back(countHammingUndetectableErrors<12>(all_stats[11]));
    all_stats.emplace_back(countHammingUndetectableErrors<13>(all_stats[12]));
    all_stats.emplace_back(countHammingUndetectableErrors<14>(all_stats[13]));
    all_stats.emplace_back(countHammingUndetectableErrors<15>(all_stats[14]));
    all_stats.emplace_back(countHammingUndetectableErrors<16>(all_stats[15]));
    all_stats.emplace_back(countHammingUndetectableErrors<17>(all_stats[16]));
    all_stats.emplace_back(countHammingUndetectableErrors<18>(all_stats[17]));
    all_stats.emplace_back(countHammingUndetectableErrors<19>(all_stats[18]));
    all_stats.emplace_back(countHammingUndetectableErrors<20>(all_stats[19]));
    all_stats.emplace_back(countHammingUndetectableErrors<21>(all_stats[20]));
    all_stats.emplace_back(countHammingUndetectableErrors<22>(all_stats[21]));
    all_stats.emplace_back(countHammingUndetectableErrors<23>(all_stats[22]));
    all_stats.emplace_back(countHammingUndetectableErrors<24>(all_stats[23]));
    all_stats.emplace_back(countHammingUndetectableErrors<25>(all_stats[24]));
    all_stats.emplace_back(countHammingUndetectableErrors<26>(all_stats[25]));
    all_stats.emplace_back(countHammingUndetectableErrors<27>(all_stats[26]));
    all_stats.emplace_back(countHammingUndetectableErrors<28>(all_stats[27]));
    all_stats.emplace_back(countHammingUndetectableErrors<29>(all_stats[28]));
    all_stats.emplace_back(countHammingUndetectableErrors<30>(all_stats[29]));
    all_stats.emplace_back(countHammingUndetectableErrors<31>(all_stats[30]));
    all_stats.emplace_back(countHammingUndetectableErrors<32>(all_stats[31]));
    all_stats.emplace_back(countHammingUndetectableErrors<33>(all_stats[32]));
    all_stats.emplace_back(countHammingUndetectableErrors<34>(all_stats[33]));
    all_stats.emplace_back(countHammingUndetectableErrors<35>(all_stats[34]));
    all_stats.emplace_back(countHammingUndetectableErrors<36>(all_stats[35]));
    all_stats.emplace_back(countHammingUndetectableErrors<37>(all_stats[36]));
    all_stats.emplace_back(countHammingUndetectableErrors<38>(all_stats[37]));
    all_stats.emplace_back(countHammingUndetectableErrors<39>(all_stats[38]));
    all_stats.emplace_back(countHammingUndetectableErrors<40>(all_stats[39]));

    return 0;
}
