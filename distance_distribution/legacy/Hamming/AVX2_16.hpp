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
 * AVX2_16.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#ifdef __AVX2__

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include "ExtHamming16.hpp"
#include "AVX2_08.hpp"

template<>
struct SIMD<__m256i, uint16_t> {
    static __m256i set1(
            uint16_t value) {
        return _mm256_set1_epi16(value);
    }

    static __m256i set_inc(
            uint16_t base,
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

extern template struct SIMD<__m256i, uint16_t>;

#endif /* __AVX2__ */
