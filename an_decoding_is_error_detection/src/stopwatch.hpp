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

#ifndef STOPWATCH_HPP__
#define STOPWATCH_HPP__

#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

class StopWatch {

    high_resolution_clock::time_point startNS, stopNS;

public:
    StopWatch ();

    void start ();
    high_resolution_clock::rep stop ();
    high_resolution_clock::rep duration ();
};

typedef struct hrc_duration {

    high_resolution_clock::rep dura;

    hrc_duration (high_resolution_clock::rep dura);
} hrc_duration;

ostream& operator<< (ostream& stream, hrc_duration hrcd);
ostream& operator<< (ostream& stream, StopWatch& sw);


#endif // STOPWATCH_HPP__
