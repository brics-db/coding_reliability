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
 * StopWatch.cpp
 *
 *  Created on: 19.09.2017
 *      Author: Till Kolditz - Till.Kolditz@gmail.com
 */

#include "StopWatch.hpp"

StopWatch::StopWatch()
        : startNS(std::chrono::nanoseconds(0)),
          stopNS(std::chrono::nanoseconds(0)) {
}

void StopWatch::start() {
    startNS = std::chrono::high_resolution_clock::now();
}

std::chrono::high_resolution_clock::rep StopWatch::stop() {
    stopNS = std::chrono::high_resolution_clock::now();
    return duration();
}

std::chrono::high_resolution_clock::rep StopWatch::duration() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stopNS - startNS).count();
}

std::ostream& operator<<(
        std::ostream& stream,
        StopWatch& sw) {
    std::chrono::high_resolution_clock::rep dura = sw.duration();
    size_t max = 1000;
    while (dura / max > 0) {
        max *= 1000;
    }
    max /= 1000;
    stream << std::setfill('0') << (dura / max);
    while (max > 1) {
        dura %= max;
        max /= 1000;
        stream << '.' << std::setw(3) << (dura / max);
    }
    stream << std::setfill(' ');
    return stream;
}
