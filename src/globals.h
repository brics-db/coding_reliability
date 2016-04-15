#ifndef GLOBALS_H_
#define GLOBALS_H_

#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

typedef unsigned uint_t;
typedef unsigned long long uintll_t;
typedef unsigned __int128 uint128_t;

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcountll __popcnt64
#endif

#define bitcount __builtin_popcountll


double get_abs_error_AN641(uint_t n, uint128_t* tgt, int offset);
double get_rel_error_AN641(uint_t n, uint128_t* tgt, int offset);

template<typename T>
T binomialCoeff(T n, T k);

class Statistics;
void process_result_hamming(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix = nullptr);
void process_result_ancoding(uint128_t* counts, Statistics stats, uint_t n, uint_t A, const char* file_prefix = nullptr);
void process_result_ancoding_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t A, uint_t iterations, const char* file_prefix = nullptr);

const uint128_t solution_an24_A641[] = {
    16777216,
           0,
    10076802,
   153434946,
  1150452598,
  6049205330,
 28899516102,
115368656122,
378189081646,
1058452761638,
2573467489292,
5466591303774,
10201679305232,
16811402697690,
24559836981782,
31900986742874,
36914263923702,
38095535515896,
35074592513254,
28799084410110,
21063110529820,
13694798923552,
7892709334408,
4016481691866,
1795726693726,
700858354732,
236678653462,
 68351026422,
 16618883926,
  3276574070,
   529150272,
    92569066,
    10129120,
     3350208,
     0
};
const uint128_t  solution_an16_A641[] = {
    65536,
        0,
        0,
   262054,
  1471362,
  5233458,
 18030526,
 51777972,
119084336,
228590406,
374383580,
526096540,
636024742,
662561746,
594903286,
460477338,
306623886,
174747162,
 84584460,
 34452696,
 11614808,
  3148728,
   651320,
   135498,
    45856,
        0
};
const uint128_t  solution_an8_A641[] = {
256,
0,
0,
596,
1302,
2316,
4624,
8218,
11060,
12318,
11024,
7378,
3958,
1792,
552,
118,
24,
0,
0};



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

#endif /* GLOBALS_H_ */
