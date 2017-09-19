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
 * ExtHamming16.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>

struct ExtHamming16 {
    typedef uint64_t accumulator_t;
    typedef uint16_t data_t;

    static const constexpr data_t pattern1 = static_cast<data_t>(0xAD5B);
    static const constexpr data_t pattern2 = static_cast<data_t>(0x366D);
    static const constexpr data_t pattern3 = static_cast<data_t>(0xC78E);
    static const constexpr data_t pattern4 = static_cast<data_t>(0x07F0);
    static const constexpr data_t pattern5 = static_cast<data_t>(0xF800);

    static data_t compute(
            const data_t value);

    static data_t compute_ext(
            const data_t value);
};
