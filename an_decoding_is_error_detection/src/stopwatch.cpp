// Copyright 2016 Till Kolditz
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

#include "stopwatch.hpp"

using namespace std;
using namespace std::chrono;

const bool OUTPUT_INSERT_DOT = true;

StopWatch::StopWatch () {
}

void
StopWatch::start () {
    startNS = high_resolution_clock::now();
}

high_resolution_clock::rep
StopWatch::stop () {
    stopNS = high_resolution_clock::now();
    return duration();
}

high_resolution_clock::rep
StopWatch::duration () {
    return duration_cast<nanoseconds>(stopNS - startNS).count();
}

hrc_duration::hrc_duration (high_resolution_clock::rep dura)
        : dura (dura) {
}

ostream& operator<< (ostream& stream, hrc_duration hrcd) {
    high_resolution_clock::rep dura = hrcd.dura;
    stringstream ss;
    if (OUTPUT_INSERT_DOT) {
        size_t max = 1000;
        while (dura / max > 0) {
            max *= 1000;
        }
        max /= 1000;
        ss << setfill('0') << (dura / max);
        while (max > 1) {
            dura %= max;
            max /= 1000;
            ss << '.' << setw(3) << (dura / max);
        }
        ss << flush;
        stream << ss.str();
    } else {
        stream << dura;
    }
    return stream;
}

ostream& operator<< (ostream& stream, StopWatch& sw) {
    return stream << hrc_duration(sw.duration());
}
