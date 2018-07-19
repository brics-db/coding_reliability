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

#ifndef GLOBALS_H_
#define GLOBALS_H_

#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <iostream>
#include <sstream>

typedef unsigned uint_t;
typedef unsigned long long uintll_t;
typedef unsigned __int128 uint128_t;

struct Flags {
  uintll_t A = 61;
  int with_grid = 0; // with "monte-carlo" grid approximation (=dim={1,2})
  int mc_iterations = 0; // number of monte carlo iterations
  int mc_iterations_2 = 0; // for 2D-grid
  uintll_t n = 16; // nr of bits as input size
  int search_super_A = 0;
  uintll_t search_start = 0;
  uintll_t search_end = 0;
  int nr_dev = 0;
  int dev = -1;
  int verbose = 0;
  const char* file_prefix = nullptr;
};

#define COUNTS_MAX_WIDTH 64

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcountll __popcnt64
#endif

#define bitcount __builtin_popcountll

template<typename T>
T binomialCoeff(T n, T k);

class Statistics;

void process_result(std::stringstream& ss,
                    uintll_t A, uint_t h,
                    const Flags& flags,
                    double* times, uint128_t* counts);

std::ostream& operator<<( std::ostream& dest, uint128_t value );

template<typename T>
T binomialCoeff(T n, T k)
{
  T res = 1;

  // Since C(n, k) = C(n, n-k)
  if ( k > n - k )
      k = n - k;

  // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
  for (T i = 0; i < k; ++i)
  {
      res *= (n - i);
      res /= (i + 1);
  }

  return res;
}


inline int getShardSize(uint_t n)
{
  if(n<16)
    return 1;
  if(n<24)
    return 128;
  if(n<32)
    return 256;
  return 512;
}

#endif /* GLOBALS_H_ */
