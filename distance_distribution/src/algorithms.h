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

#ifndef ALGORITHMS_H_
#define ALGORITHMS_H_

#include "globals.h"

void run_hamming(uintll_t,int,int,int);
void run_hamming_cpu(uintll_t,int,int);
void run_hamming_cpu_native(uintll_t,int,int);
void run_hamming_cpu_native_short(uintll_t,int,int);
void run_hamming_cpu_native_short_ordering(uintll_t,int,int);
void run_ancoding(uintll_t, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb,int,int);
void run_ancoding_cpu(uintll_t, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb,int);
double run_hamming_mc(uintll_t,int,uintll_t,int,int);
double run_ancoding_mc(uintll_t, uintll_t, uintll_t A, int, double*, uintll_t* minb, uint128_t* mincb,int,int);
double run_ancoding_grid(int,uintll_t, uintll_t, uintll_t, uintll_t A, int, double*, uintll_t* minb, uint128_t* mincb,int,int);
double run_hamming_grid(uintll_t, int, uintll_t, int, int);
//double run_ancoding_mc_v2(uintll_t, uintll_t, int);
void test_curand(uintll_t n, uintll_t iterations,int);
void test_curand_1d(uintll_t n, uintll_t iterations,int);
void test_curand_raw(uintll_t n, uintll_t iterations,int);

#endif
