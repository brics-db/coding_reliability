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
