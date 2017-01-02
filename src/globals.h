#ifndef GLOBALS_H_
#define GLOBALS_H_

#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <iostream>

typedef unsigned uint_t;
typedef unsigned long long uintll_t;
typedef unsigned __int128 uint128_t;

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcountll __popcnt64
#endif

#define bitcount __builtin_popcountll

double get_rel_error_hamming(uint_t n, uint128_t* tgt, int offset, int w1bit, double* errors=nullptr);
double get_abs_error_hamming(uint_t n, uint128_t* tgt, int offset, int w1bit, double* errors=nullptr);
double get_abs_error_AN(uintll_t A, uint_t n, uint128_t* tgt, int offset, double* errors=nullptr);
double get_rel_error_AN(uintll_t A, uint_t n, uint128_t* tgt, int offset, double* errors=nullptr);

template<typename T>
T binomialCoeff(T n, T k);

class Statistics;
void process_result_hamming(uint128_t* counts, Statistics stats, uint_t n, uint_t h, const char* file_prefix = nullptr);
void process_result_hamming_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t h, int, uint_t iterations, const char* file_prefix = nullptr);
void process_result_ancoding(uint128_t* counts, Statistics stats, uint_t n, uint_t A, const char* file_prefix = nullptr, std::string app = "");
void process_result_ancoding_mc(uint128_t* counts, Statistics stats, uint_t n, uint_t A, uint_t iterations, const char* file_prefix = nullptr, std::string app = "");
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
