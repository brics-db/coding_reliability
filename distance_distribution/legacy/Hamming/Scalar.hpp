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
 * Scalar.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>
#include <exception>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt64
#define __builtin_popcountl __popcnt64
#define __builtin_popcountll __popcnt64
#endif

#include "ExtHamming.hpp"

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

extern template struct Scalar<uint8_t> ;
extern template struct Scalar<uint16_t> ;
extern template struct Scalar<uint32_t> ;
extern template struct Scalar<uint64_t> ;
