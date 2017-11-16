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
 * pascals_triangle.cpp
 *
 *  Created on: 15.11.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include <iostream>
#include <exception>
#include <cstdlib>

#include <boost/multiprecision/cpp_int.hpp>

void usage(
        int argc,
        char ** argv) {
    std::cerr << "Usage: " << argv[0] << " <number of rows>" << std::endl;
}

template<typename T>
T factorial(
        T n) {
    T result = T(1);
    for (; n; --n)
        result *= n;
    return result;
}

template<typename T>
T binomial_coefficient(
        T n,
        T k) {
    T result = 1;
    if (n < k)
        throw std::runtime_error("n < k");
    if (k > 1) {
        for (T i = T(1); i <= k; ++i) {
            result = result * (n + 1 - i) / i;
        }
    }
    return result;
}

template<typename T>
void print_layer(
        size_t numCurrentRow) {
    T number = 1;
    for (size_t j = 0; j <= numCurrentRow; j++) {
        std::cout << number << ' ';
        number = number * (numCurrentRow - j) / (j + 1);
    }
}

int main(
        int argc,
        char ** argv) {
    if (argc != 2) {
        usage(argc, argv);
        return 1;
    }

    char * str_end;
    size_t numRows = std::strtoull(argv[1], &str_end, 10);

    for (size_t numCurrentRow = 0; numCurrentRow < numRows; numCurrentRow++) {
        for (size_t k = numRows; k > numCurrentRow; k--) {
            std::cout << ' ';
        }
        if (numCurrentRow < 32) {
            print_layer<uint32_t>(numCurrentRow);
        } else if (numCurrentRow < 64) {
            print_layer<uint64_t>(numCurrentRow);
        } else if (numCurrentRow < 128) {
            print_layer<boost::multiprecision::uint128_t>(numCurrentRow);
        }
        std::cout << '\n';
    }
    return 0;
}
