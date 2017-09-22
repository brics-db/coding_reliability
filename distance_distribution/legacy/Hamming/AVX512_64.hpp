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
 * AVX512_64.hpp
 *
 *  Created on: 22.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#ifdef __AVX512F__

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

#include "SIMD.hpp"
#include "ExtHamming64.hpp"

template<>
struct SIMD<__m512i, uint64_t> {
    static __m512i set1(
            uint64_t value) {
        return _mm512_set1_epi64(value);
    }

    static __m512i set_inc(
            uint64_t base,
            uint64_t inc = 1) {
        return _mm512_set_epi64(base + 7 * inc, base + 6 * inc, base + 5 * inc, base + 4 * inc, base + 3 * inc, base + 2 * inc, base + 1 * inc, base);
    }

    static __m512i add(
            __m512i a,
            __m512i b) {
        return _mm512_add_epi64(a, b);
    }

    static __m512i popcount(
            __m512i data) {
        auto pattern1 = _mm512_set1_epi64(0x5555555555555555);
        auto pattern2 = _mm512_set1_epi64(0x3333333333333333);
        auto pattern3 = _mm512_set1_epi64(0x0F0F0F0F0F0F0F0F);
        auto pattern4 = _mm512_set1_epi64(0x0101010101010101);
        auto temp = _mm512_sub_epi64(data, _mm512_and_si512(_mm512_srli_epi64(data, 1), pattern1));
        temp = _mm512_add_epi64(_mm512_and_si512(temp, pattern2), _mm512_and_si512(_mm512_srli_epi64(temp, 2), pattern2));
        temp = _mm512_and_si512(_mm512_add_epi64(temp, _mm512_srli_epi64(temp, 4)), pattern3);
        auto tmp2 = _mm512_mul_epu32(temp, pattern4);
        auto tmp3 = _mm512_mul_epu32(_mm512_srli_epi64(temp, 32), pattern4);
        temp = _mm512_add_epi64(tmp2, _mm512_slli_epi64(tmp3, 32));
        return _mm512_srli_epi64(temp, 56);
    }

#define SET_RESULT(vec, result) do { \
    auto vec2 = reinterpret_cast<uint64_t*>(&vec); \
    result[0] = static_cast<uint8_t>(vec2[0]); \
    result[1] = static_cast<uint8_t>(vec2[1]); \
    result[2] = static_cast<uint8_t>(vec2[2]); \
    result[3] = static_cast<uint8_t>(vec2[3]); \
    result[4] = static_cast<uint8_t>(vec2[4]); \
    result[5] = static_cast<uint8_t>(vec2[5]); \
    result[6] = static_cast<uint8_t>(vec2[6]); \
    result[7] = static_cast<uint8_t>(vec2[7]); \
} while (false)

    // returs the total population count of both the data and hamming code bits
    static uint64_t count_hamming(
            __m512i data) {
        auto pattern1 = _mm512_set1_epi64(ExtHamming64::pattern1);
        auto pattern2 = _mm512_set1_epi64(ExtHamming64::pattern2);
        auto pattern3 = _mm512_set1_epi64(ExtHamming64::pattern3);
        auto pattern4 = _mm512_set1_epi64(ExtHamming64::pattern4);
        auto pattern5 = _mm512_set1_epi64(ExtHamming64::pattern5);
        auto pattern6 = _mm512_set1_epi64(ExtHamming64::pattern6);
        auto pattern7 = _mm512_set1_epi64(ExtHamming64::pattern7);
        auto mask = _mm512_set1_epi64(0x1);
        uint8_t result[8];
        auto accumulator = popcount(data);
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern1)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern2)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern3)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern4)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern5)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern6)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern7)), mask));
        SET_RESULT(accumulator, result);
        return reinterpret_cast<uint64_t*>(result)[0];
    }

    // returs the total population count of both the data and extended hamming code bits
    static uint64_t count_ext_hamming(
            __m512i data) {
        auto pattern1 = _mm512_set1_epi64(ExtHamming64::pattern1);
        auto pattern2 = _mm512_set1_epi64(ExtHamming64::pattern2);
        auto pattern3 = _mm512_set1_epi64(ExtHamming64::pattern3);
        auto pattern4 = _mm512_set1_epi64(ExtHamming64::pattern4);
        auto pattern5 = _mm512_set1_epi64(ExtHamming64::pattern5);
        auto pattern6 = _mm512_set1_epi64(ExtHamming64::pattern6);
        auto pattern7 = _mm512_set1_epi64(ExtHamming64::pattern7);
        auto mask = _mm512_set1_epi64(0x1);
        uint8_t result[8];
        auto accumulator = popcount(data);
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(accumulator, mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern1)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern2)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern3)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern4)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern5)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern6)), mask));
        accumulator = _mm512_add_epi64(accumulator, _mm512_and_si512(popcount(_mm512_and_si512(data, pattern7)), mask));
        SET_RESULT(accumulator, result);
        return reinterpret_cast<uint64_t*>(result)[0];
    }
};

#endif /* __AVX512F__ */
