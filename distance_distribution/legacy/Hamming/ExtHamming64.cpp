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
 * ExtHamming64.cpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include "Scalar.hpp"

typedef typename ExtHamming64::data_t data_t;

const data_t ExtHamming64::pattern1;
const data_t ExtHamming64::pattern2;
const data_t ExtHamming64::pattern3;
const data_t ExtHamming64::pattern4;
const data_t ExtHamming64::pattern5;
const data_t ExtHamming64::pattern6;
const data_t ExtHamming64::pattern7;

size_t ExtHamming64::compute(
        const size_t value) {
    size_t hamming = 0;
    hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1);
    hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 1;
    hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 2;
    hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 3;
    hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 4;
    hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 5;
    hamming |= (Scalar<data_t>::popcount(value & pattern7) & 0x1) << 6;
    return hamming;
}

size_t ExtHamming64::compute_ext(
        const size_t value) {
    size_t hamming = 0;
    hamming |= (Scalar<data_t>::popcount(value & pattern1) & 0x1) << 1;
    hamming |= (Scalar<data_t>::popcount(value & pattern2) & 0x1) << 2;
    hamming |= (Scalar<data_t>::popcount(value & pattern3) & 0x1) << 3;
    hamming |= (Scalar<data_t>::popcount(value & pattern4) & 0x1) << 4;
    hamming |= (Scalar<data_t>::popcount(value & pattern5) & 0x1) << 5;
    hamming |= (Scalar<data_t>::popcount(value & pattern6) & 0x1) << 6;
    hamming |= (Scalar<data_t>::popcount(value & pattern7) & 0x1) << 7;
    hamming |= (Scalar<data_t>::popcount(value) + Scalar<data_t>::popcount(hamming)) & 0x1;
    return hamming;
}
