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
 * Statistics.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <utility>
#include <boost/multiprecision/cpp_int.hpp>

struct Statistics {
    typedef boost::multiprecision::uint256_t counts_t;

    const size_t numDataBits;
    const size_t numHammingBits;
    const size_t numCodeBits;
    counts_t numCodeWords;
    const size_t numCounts;
    std::unique_ptr<counts_t[]> counts;

    Statistics(
            size_t numDataBits,
            size_t numHammingBits);

    Statistics(
            size_t numDataBits,
            size_t numHammingBits,
            counts_t * counts);

    Statistics(
            size_t numDataBits,
            size_t numHammingBits,
            uint64_t * counts);

    Statistics(
            size_t numDataBits,
            size_t numHammingBits,
            uint32_t * counts);

    Statistics(
            size_t numDataBits,
            size_t numHammingBits,
            uint16_t * counts);

    Statistics(
            size_t numDataBits,
            size_t numHammingBits,
            uint8_t * counts);

    Statistics(
            Statistics && other);

    Statistics(
            Statistics & other) = delete;

    Statistics & operator=(
            Statistics && other);

    Statistics & operator=(
            Statistics & other) = delete;
};
