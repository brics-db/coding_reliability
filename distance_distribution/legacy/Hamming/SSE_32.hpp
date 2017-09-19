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
 * SSE_32.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#ifdef __SSE4_2__

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include "ExtHamming32.hpp"
#include "SSE_08.hpp"

template<>
struct SIMD<__m128i, uint32_t> {
    static __m128i set1(
            uint32_t value) {
        return _mm_set1_epi32(value);
    }

    static __m128i set_inc(
            uint32_t base,
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

extern template struct SIMD<__m128i, uint32_t>;

#endif /* __SSE4_2__ */
