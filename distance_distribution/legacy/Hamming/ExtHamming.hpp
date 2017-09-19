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
 * ExtHamming.hpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "ExtHamming08.hpp"
#include "ExtHamming16.hpp"
#include "ExtHamming32.hpp"
#include "ExtHamming64.hpp"

template<size_t BITCNT_DATA>
struct ExtHamming {
    static const constexpr size_t BITCNT_HAMMING = (
            BITCNT_DATA == 1 ? 3 : (BITCNT_DATA <= 4 ? 4 : (BITCNT_DATA <= 11 ? 5 : (BITCNT_DATA <= 26 ? 6 : (BITCNT_DATA <= 57 ? 7 : BITCNT_DATA <= 64 ? 8 : 9)))));

    typedef typename std::conditional<BITCNT_DATA <= 8, ExtHamming08,
            typename std::conditional<BITCNT_DATA <= 16, ExtHamming16,
                    typename std::conditional<BITCNT_DATA <= 32, ExtHamming32, typename std::conditional<BITCNT_DATA <= 64, ExtHamming64, void>::type>::type>::type>::type hamming_impl_t;
    typedef typename hamming_impl_t::data_t data_t;
};
