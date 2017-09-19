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
 * ExtHamming32.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>

struct ExtHamming32 {
    typedef typename boost::multiprecision::uint128_t accumulator_t;
    typedef uint32_t data_t;

    static const constexpr data_t pattern1 = 0x56AAAD5B;
    static const constexpr data_t pattern2 = 0x9B33366D;
    static const constexpr data_t pattern3 = 0xE3C3C78E;
    static const constexpr data_t pattern4 = 0x03FC07F0;
    static const constexpr data_t pattern5 = 0x03FFF800;
    static const constexpr data_t pattern6 = 0xFC000000;

    static data_t compute(
            const data_t value);

    static data_t compute_ext(
            const data_t value);
};
