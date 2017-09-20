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
 * Statistics.cpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include "Statistics.hpp"

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = 0;
    }
}

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits,
        counts_t * counts)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = counts[i];
    }
}

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits,
        uint64_t * counts)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = counts[i];
    }
}

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits,
        uint32_t * counts)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = counts[i];
    }
}

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits,
        uint16_t * counts)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = counts[i];
    }
}

Statistics::Statistics(
        size_t numDataBits,
        size_t numHammingBits,
        uint8_t * counts)
        : numDataBits(numDataBits),
          numHammingBits(numHammingBits),
          numCodeBits(numDataBits + numHammingBits),
          numCodeWords(1),
          numCounts(numCodeBits + 1),
          counts(new counts_t[numCounts]) {
    numCodeWords <<= numDataBits;
    for (size_t i = 0; i < numCounts; ++i) {
        this->counts[i] = counts[i];
    }
}

Statistics::Statistics(
        Statistics && other)
        : numDataBits(other.numDataBits),
          numHammingBits(other.numHammingBits),
          numCodeBits(other.numCodeBits),
          numCodeWords(other.numCodeWords),
          numCounts(other.numCounts),
          counts(std::move(other.counts)) {
}

Statistics & Statistics::operator=(
        Statistics && other) {
    new (this) Statistics(std::forward<Statistics>(other));
    return *this;
}
