// Copyright 2017 Till Kolditz
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

/*
 * AVX2_08.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#ifdef __AVX2__

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include "SIMD.hpp"
#include "ExtHamming08.hpp"

template<>
struct SIMD<__m256i, uint8_t> {
    static __m256i set1(
            uint8_t value) {
        return _mm256_set1_epi8(value);
    }

    static __m256i set_inc(
            uint8_t base,
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

extern template struct SIMD<__m256i, uint8_t>;

#endif /* __AVX2__ */
