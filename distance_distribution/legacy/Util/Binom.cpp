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
 * Binom.cpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include <boost/multiprecision/cpp_int.hpp>

#include "Binom.hpp"

/**
 * Taken and adapted from: https://gist.github.com/jeetsukumaran/5392166
 *
 * Calculates the binomial coefficient, $\choose{n, k}$, i.e., the number of
 * distinct sets of $k$ elements that can be sampled with replacement from a
 * population of $n$ elements.
 *
 * @tparam T
 *   Numeric type. Defaults to unsigned long.
 * @param n
 *   Population size.
 * @param k
 *   Number of elements to sample without replacement.
 *
 * @return
 *   The binomial coefficient, $\choose{n, k}$.
 *
 * @note
 *    Modified from: http://etceterology.com/fast-binomial-coefficients
 */
template<class T = unsigned long>
T binomial_coefficient(
        size_t n,
        size_t k) {
    unsigned long i;
    if (0 == k || n == k) {
        return 1;
    }
    if (k > n) {
        return 0;
    }
    if (k > (n - k)) {
        k = n - k;
    }
    if (1 == k) {
        return n;
    }
    T b = 1;
    for (i = 1; i <= k; ++i) {
        b *= (n - (k - i));
        if (b < 0)
            return -1; /* Overflow */
        b /= i;
    }
    return b;
}

template uint16_t binomial_coefficient<uint16_t>(
        size_t n,
        size_t k);

template uint32_t binomial_coefficient<uint32_t>(
        size_t n,
        size_t k);

template uint64_t binomial_coefficient<uint64_t>(
        size_t n,
        size_t k);

template boost::multiprecision::uint128_t binomial_coefficient<boost::multiprecision::uint128_t>(
        size_t n,
        size_t k);

template boost::multiprecision::uint256_t binomial_coefficient<boost::multiprecision::uint256_t>(
        size_t n,
        size_t k);

template boost::multiprecision::uint512_t binomial_coefficient<boost::multiprecision::uint512_t>(
        size_t n,
        size_t k);

template boost::multiprecision::uint1024_t binomial_coefficient<boost::multiprecision::uint1024_t>(
        size_t n,
        size_t k);
