// Copyright 2016 Matthias Werner
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

#ifndef TIMER_H_
#define TIMER_H_

#include <cuda_runtime.h>

enum TimeType { CPU_WALL_TIME, CPU_CLOCK_TIME, GPU_TIME };
struct MTime {
    double start, time;
    TimeType type;
    cudaEvent_t gpustart, gpustop;
    MTime():start(0.0),time(0.0),type(CPU_WALL_TIME),gpustart(0),gpustop(0){}
};

void startTimer_CPU( MTime* );
void startTimer_CPUWall( MTime* );
double stopTimer( MTime* );

void startTimer_GPU( MTime* );


#endif /* TIMER_H_ */
