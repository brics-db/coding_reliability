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
 * SSE_08.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#ifdef __SSE4_2__

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include "SIMD.hpp"
#include "ExtHamming08.hpp"

template<>
struct SIMD<__m128i, uint8_t> {
    static __m128i set1(
            uint8_t value) {
        return _mm_set1_epi8(value);
    }

    static __m128i set_inc(
            uint8_t base,
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

extern template struct SIMD<__m128i, uint8_t>;

#endif /* __SSE4_2__ */
