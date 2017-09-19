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
 * ExtHamming32.cpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include "Scalar.hpp"

typedef typename ExtHamming32::data_t data_t;

const data_t ExtHamming32::pattern1;
const data_t ExtHamming32::pattern2;
const data_t ExtHamming32::pattern3;
const data_t ExtHamming32::pattern4;
const data_t ExtHamming32::pattern5;
const data_t ExtHamming32::pattern6;

data_t ExtHamming32::compute(
        const data_t value) {
    data_t hamming = 0;
    hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
    hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
    hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
    hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
    hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 4;
    hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 5;
    return hamming;
}

data_t ExtHamming32::compute_ext(
        const data_t value) {
    data_t hamming = 0;
    hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
    hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
    hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
    hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
    hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 5;
    hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 6;
    hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
    return hamming;
}
