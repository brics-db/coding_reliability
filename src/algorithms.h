#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

#include "globals.h"

void run_hamming(uintll_t,int,int,int);
void run_hamming_cpu(uintll_t,int,int);
void run_hamming_cpu_native(uintll_t,int,int);
void run_hamming_cpu_native_short(uintll_t,int,int);
void run_hamming_cpu_native_short_ordering(uintll_t,int,int);
void run_ancoding(uintll_t, uintll_t A, int verbose, uintll_t* minb, uintll_t* mincb,int,int);
void run_ancoding_cpu(uintll_t, uintll_t A, int verbose, uintll_t* minb, uintll_t* mincb,int);
double run_hamming_mc(uintll_t,int,uintll_t,int,int);
double run_ancoding_mc(uintll_t, uintll_t, uintll_t A, int, double*, uintll_t* minb, uintll_t* mincb,int,int);
//double run_ancoding_mc_v2(uintll_t, uintll_t, int);
void test_curand(uintll_t n, uintll_t iterations);
void test_curand_1d(uintll_t n, uintll_t iterations);
void test_curand_raw(uintll_t n, uintll_t iterations);

#endif
