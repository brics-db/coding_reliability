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
#include <omp.h>
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt64
#endif

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

    static __m128i hamming(
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
        auto mask = _mm_set1_epi16(0x0101);
        auto shuffle = _mm_set_epi64x(0xFF0FFF0DFF0BFF09, 0xFF07FF05FF03FF01);
        auto popcount8 = SIMD<__m128i, uint8_t>::popcount(a);
        return _mm_shuffle_epi8(_mm_mullo_epi16(popcount8, mask), shuffle);
    }

    static __m128i hamming(
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
        auto mask = _mm_set1_epi32(0x01010101);
        auto shuffle = _mm_set_epi64x(0xFFFFFF0FFFFFFF0B, 0xFFFFFF07FFFFFF03);
        auto popcount8 = SIMD<__m128i, uint8_t>::popcount(a);
        return _mm_shuffle_epi8(_mm_mullo_epi16(popcount8, mask), shuffle);
    }

    static __m128i hamming(
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
        auto mask = _mm_set1_epi64x(0x0101010101010101);
        auto shuffle = _mm_set_epi64x(0xFFFFFFFFFFFFFF0F, 0xFFFFFFFFFFFFFF07);
        auto popcount8 = SIMD<__m128i, uint8_t>::popcount(a);
        return _mm_shuffle_epi8(_mm_mullo_epi16(popcount8, mask), shuffle);
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

    static __m256i hamming(
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
        auto mask = _mm256_set1_epi32(0x01010101);
        auto shuffle = _mm256_set_epi64x(0xFFFFFF0FFFFFFF0B, 0xFFFFFF07FFFFFF03, 0xFFFFFF0FFFFFFF0B, 0xFFFFFF07FFFFFF03);
        auto popcount8 = SIMD<__m256i, uint8_t>::popcount(a);
        return _mm256_shuffle_epi8(_mm256_mullo_epi16(popcount8, mask), shuffle);
    }

    static __m256i hamming(
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
        tmp1 = _mm256_and_si256(popcount(_mm256_and_si256(data, pattern6)), mask);
        tmp2 = _mm256_xor_si256(tmp2, tmp1);
        codebits = _mm256_or_si256(codebits, _mm256_slli_epi16(tmp1, ++shift));
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
        auto mask = _mm256_set1_epi64x(0x0101010101010101);
        auto shuffle = _mm256_set_epi64x(0xFFFFFFFFFFFFFF0F, 0xFFFFFFFFFFFFFF07, 0xFFFFFFFFFFFFFF0F, 0xFFFFFFFFFFFFFF07);
        auto popcount8 = SIMD<__m256i, uint8_t>::popcount(a);
        return _mm256_shuffle_epi8(_mm256_mullo_epi16(popcount8, mask), shuffle);
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

struct Hamming08 {
    typedef size_t accumulator_t;
    typedef uint8_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 8;
    static const constexpr size_t BITCNT_HAMMING = 5;

    static inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x0000005B) & 0x1) << 1;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x0000006D) & 0x1) << 2;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x0000008E) & 0x1) << 3;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x000000F0) & 0x1) << 4;
        hamming |= (Scalar<uint8_t>::popcount(value & 0x000000FF) + Scalar<uint8_t>::popcount(hamming)) & 0x1;
        return (value << 5) | hamming;
    }
};

struct Hamming16 {
    typedef size_t accumulator_t;
    typedef uint16_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 16;
    static const constexpr size_t BITCNT_HAMMING = 6;

    inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x0000AD5B) & 0x1) << 1;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x0000366D) & 0x1) << 2;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x0000C78E) & 0x1) << 3;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x000007F0) & 0x1) << 4;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x0000F800) & 0x1) << 5;
        hamming |= (Scalar<uint16_t>::popcount(value & 0x0000FFFF) + Scalar<uint16_t>::popcount(hamming)) & 0x1;
        return (value << 6) | hamming;
    }
};

struct Hamming24 {
    typedef size_t accumulator_t;
    typedef uint32_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 24;
    static const constexpr size_t BITCNT_HAMMING = 6;

    inline size_t compute(
            const size_t value) {
        size_t hamming = 0;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x00AAAD5B) & 0x1) << 1;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x0033366D) & 0x1) << 2;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x00C3C78E) & 0x1) << 3;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x00FC07F0) & 0x1) << 4;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x00FFF800) & 0x1) << 5;
        hamming |= (Scalar<uint32_t>::popcount(value & 0x00FFFFFF) + Scalar<uint32_t>::popcount(hamming)) & 0x1;
        return (value << 6) | hamming;
    }
};

struct Hamming32 {
    typedef size_t accumulator_t;
    typedef uint32_t popcount_t;

    static const constexpr size_t BITCNT_DATA = 32;
    static const constexpr size_t BITCNT_HAMMING = 7;

    inline size_t compute(
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

template<typename T, size_t ACTUAL_BITCNT_DATA = T::BITCNT_DATA, size_t ACTUAL_BITCNT_HAMMING = T::BITCNT_HAMMING>
void countHammingUndetectableErrors() {
    const constexpr size_t BITCNT_CODEWORD = ACTUAL_BITCNT_DATA + ACTUAL_BITCNT_HAMMING;
    const constexpr size_t CNT_COUNTS = BITCNT_CODEWORD + 1ull;

#ifndef NDEBUG
    std::cout << "BITCNT_CODEWORD=" << BITCNT_CODEWORD << '\n';
    std::cout << "CNT_COUNTS=" << CNT_COUNTS << '\n';
#pragma omp parallel
    {
        const size_t num_threads = omp_get_num_threads();
#pragma omp master
        {
            std::cout << "OpenMP using " << num_threads << " threads." << std::endl;
        }
    }
#endif

    typedef typename T::accumulator_t accumulator_t;
    typedef typename T::popcount_t popcount_t;

#if defined (_MSC_VER) and not defined (size_t)
    typedef __int64 size_t;
#endif

    StopWatch sw;
    sw.start();

    accumulator_t * counts = new accumulator_t[CNT_COUNTS]();

#pragma omp parallel shared(counts)
    {
        const size_t num_threads = omp_get_num_threads();
        const size_t bucket_size = (0x1 << ACTUAL_BITCNT_DATA) / num_threads;
        const size_t remaining = (0x1 << ACTUAL_BITCNT_DATA) % num_threads;

#pragma omp for schedule(dynamic,1)
        for (size_t thread = 0; thread < num_threads; ++thread) {
            accumulator_t * counts_local = new accumulator_t[CNT_COUNTS]();
            size_t x = bucket_size * thread;
            const size_t max = x + bucket_size + (thread == (num_threads - 1) ? remaining : 0); // the last thread does the few additional remaining code words

#ifdef __AVX2__
            const constexpr size_t values_per_mm256 = sizeof(__m256i) / sizeof(popcount_t);
            auto mm256 = SIMD<__m256i, popcount_t>::set_inc(x, 1);
            auto mm256inc = SIMD<__m256i, popcount_t>::set1(values_per_mm256);

            for (; x <= (max - values_per_mm256); x += values_per_mm256) {
                auto popcount = SIMD<__m256i, popcount_t>::popcount(mm256);
                auto hamming = SIMD<__m256i, popcount_t>::hamming(mm256);
                auto hammingPopcount = SIMD<__m256i, popcount_t>::popcount(hamming);
                popcount = SIMD<__m256i, popcount_t>::add(popcount, hammingPopcount);
                popcount_t * pPopcount = reinterpret_cast<popcount_t*>(&popcount);
                for (size_t i = 0; i < values_per_mm256; ++i) {
                    counts_local[pPopcount[i]]++;
                }
                mm256 = SIMD<__m256i, popcount_t>::add(mm256, mm256inc);
            }
#endif /* __AVX2__ */

// #ifdef __SSE4_2__
            const constexpr size_t values_per_mm128 = sizeof(__m128i) / sizeof(popcount_t);
            auto mm128 = SIMD<__m128i, popcount_t>::set_inc(x, 1);
            auto mm128inc = SIMD<__m128i, popcount_t>::set1(values_per_mm128);

            for (; x <= (max - values_per_mm128); x += values_per_mm128) {
                auto popcount = SIMD<__m128i, popcount_t>::popcount(mm128);
                auto hamming = SIMD<__m128i, popcount_t>::hamming(mm128);
                auto hammingPopcount = SIMD<__m128i, popcount_t>::popcount(hamming);
                popcount = SIMD<__m128i, popcount_t>::add(popcount, hammingPopcount);
                popcount_t * pPopcount = reinterpret_cast<popcount_t*>(&popcount);
                for (size_t i = 0; i < values_per_mm128; ++i) {
                    counts_local[pPopcount[i]]++;
                }
                mm128 = SIMD<__m128i, popcount_t>::add(mm128, mm128inc);
            }
// #endif /* __SSE4_2__ */

            for (; x < max; ++x) {
                counts_local[__builtin_popcountll(T::compute(x))]++;
            }

            // 4) Sum the counts
            for (size_t i = 0; i < CNT_COUNTS; ++i) {
#pragma omp atomic
                counts[i] += counts_local[i];
            }
        }
    }
    sw.stop();

    accumulator_t * act_counts = new accumulator_t[CNT_COUNTS]();
    size_t numCodeWords = 0x1 << ACTUAL_BITCNT_DATA;

    // the transitions apply to all code words
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        act_counts[i] = counts[i] * numCodeWords;
    }

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
    size_t maxWidth = 0;
    do {
        max /= 10;
        ++maxWidth;
    } while (max);
    size_t numTotal = 0;
    size_t numTransitions = 0;
    std::cout << std::dec << "#Distances:\n";
    for (size_t i = 0; i < CNT_COUNTS; ++i) {
        numTotal += counts[i];
        numTransitions += act_counts[i];
        std::cout << "  " << std::right << std::setw(3) << i << ": " << std::setw(maxWidth) << act_counts[i] << '\n';
    }

    std::cout << "Computation took " << sw << "ns for " << numTotal << " code words and " << numTransitions << " transitions." << std::endl;
}

int main() {
    countHammingUndetectableErrors<Hamming08>();
    // countHammingUndetectableErrors<Hamming16>();
    // countHammingUndetectableErrors<Hamming24>();
    // countHammingUndetectableErrors<Hamming32>();

    return 0;
}
