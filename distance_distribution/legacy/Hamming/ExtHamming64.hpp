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
 * ExtHamming64.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>

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

    static size_t compute(
            const size_t value);

    static size_t compute_ext(
            const size_t value);
};
