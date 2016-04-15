#include "globals.h"
#include "algorithms.h"
#include "helper.h"

#include <omp.h>
#include <math.h>
#include <iostream>
#include <string.h>
using namespace std;

inline uintll_t computeHamming08(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (__builtin_popcount(value & 0x0000005B) & 0x1) << 1;
  hamming |= (__builtin_popcount(value & 0x0000006D) & 0x1) << 2;
  hamming |= (__builtin_popcount(value & 0x0000008E) & 0x1) << 3;
  hamming |= (__builtin_popcount(value & 0x000000F0) & 0x1) << 4;
  hamming |= (__builtin_popcount(value & 0x000000FF) + __builtin_popcount(hamming)) & 0x1;
  return (value << 5) | hamming;
}

inline uintll_t computeHamming16(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (__builtin_popcount(value & 0x0000AD5B) & 0x1) << 1;
  hamming |= (__builtin_popcount(value & 0x0000366D) & 0x1) << 2;
  hamming |= (__builtin_popcount(value & 0x0000C78E) & 0x1) << 3;
  hamming |= (__builtin_popcount(value & 0x000007F0) & 0x1) << 4;
  hamming |= (__builtin_popcount(value & 0x0000F800) & 0x1) << 5;
  hamming |= (__builtin_popcount(value & 0x0000FFFF) + __builtin_popcount(hamming)) & 0x1;
  return (value << 6) | hamming;
}

inline uintll_t computeHamming24(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (__builtin_popcount(value & 0x00AAAD5B) & 0x1) << 1;
  hamming |= (__builtin_popcount(value & 0x0033366D) & 0x1) << 2;
  hamming |= (__builtin_popcount(value & 0x00C3C78E) & 0x1) << 3;
  hamming |= (__builtin_popcount(value & 0x00FC07F0) & 0x1) << 4;
  hamming |= (__builtin_popcount(value & 0x00FFF800) & 0x1) << 5;
  hamming |= (__builtin_popcount(value & 0x00FFFFFF) + __builtin_popcount(hamming)) & 0x1;
  return (value << 6) | hamming;
}

inline uintll_t computeHamming32(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (__builtin_popcount(value & 0x56AAAD5B) & 0x1) << 1;
  hamming |= (__builtin_popcount(value & 0x9B33366D) & 0x1) << 2;
  hamming |= (__builtin_popcount(value & 0xE3C3C78E) & 0x1) << 3;
  hamming |= (__builtin_popcount(value & 0x03FC07F0) & 0x1) << 4;
  hamming |= (__builtin_popcount(value & 0x03FFF800) & 0x1) << 5;
  hamming |= (__builtin_popcount(value & 0xFC000000) & 0x1) << 6;
  hamming |= (__builtin_popcount(value & 0xFFFFFFFF) + __builtin_popcount(hamming)) & 0x1;
  return (value << 7) | hamming;
}

template<typename T>
inline uintll_t computeDistance(const T &value1, const T &value2) {
  return static_cast<uintll_t>(__builtin_popcount(value1 ^ value2));
}

typedef uintll_t (*computeHamming_ft)(const uintll_t &);

template<typename T, 
         uintll_t BITCNT_DATA, 
         computeHamming_ft func, 
         bool WITH_1BIT,
         uintll_t BITCNT_HAMMING = (BITCNT_DATA == 8 ? 5 : ((BITCNT_DATA == 16) | (BITCNT_DATA == 24) ? 6 : 7)), 
         uintll_t BITCNT_MSG = BITCNT_DATA + BITCNT_HAMMING, 
         uintll_t CNT_COUNTS = BITCNT_MSG + 1ull, 
         uintll_t CNT_MESSAGES = 0x1ull << BITCNT_DATA,
         uintll_t CNT_WORDS = 1ull<<BITCNT_MSG                            
>
void countHammingUndetectableErrors(uint128_t* result_counts) 
{
  T counts[ CNT_COUNTS ] = {0};
#pragma omp parallel
  {
    T counts_local[ CNT_COUNTS ] = {0};
    T x,y;
    T distance;
/*
#pragma omp master
    {
      cout << "OpenMP using " << omp_get_num_threads() << " threads" << endl;
    }*/
#pragma omp for schedule(static)
    for(T a=0; a<CNT_MESSAGES; ++a)
    {
      memset(counts_local, 0, CNT_COUNTS*sizeof(T));
      x = func(a);
      // valid codewords transitions
      for(T b=a+1; b<CNT_MESSAGES; ++b)
      {
        y = func(b);
        distance = computeDistance(x,y);
        ++counts_local[distance];
        if(WITH_1BIT)
        {
          for(T p=0;p<BITCNT_MSG;++p)
          {
            distance=computeDistance(x,y^(static_cast<T>(1)<<p));
            ++counts_local[distance];
          }
        }
      }
      // 4) Sum the counts
      for (uint_t i = 0; i < CNT_COUNTS; ++i) {
#pragma omp atomic
        counts[i] += counts_local[i];
      }      
    }
  }
  result_counts[0] = 1ull<<BITCNT_DATA;
  result_counts[1] = (BITCNT_MSG)*result_counts[0];
  result_counts[2] = 0;
  for(uint_t i=3;i<CNT_COUNTS; ++i){
    result_counts[i] = static_cast<uint128_t>(counts[i])<<1;
    //printf("%u %12llu\n", i, counts[i]);
  }
}

extern void run_hamming_cpu_native_short_ordering(uintll_t n, int with_1bit, int file_output)
{
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  
  const uintll_t h = ( n==8 ? 5 : (n<32?6:7) );
  uint128_t* counts = new uint128_t[n+h+1];
  memset(counts, 0, (n+h+1)*sizeof(uint128_t));

  printf("Start Hamming Coding Algorithm - Native Short Approach with Ordering (CPU)\n");
  results_cpu.start(i_totaltime);

  if(with_1bit)
  {
    if(n==8)
      countHammingUndetectableErrors<uintll_t, 8, computeHamming08,true>(counts);
    else if(n==16)
      countHammingUndetectableErrors<uintll_t, 16, computeHamming16,true>(counts);
    else
      countHammingUndetectableErrors<uintll_t, 24, computeHamming24,true>(counts);
  }else{
    if(n==8)
      countHammingUndetectableErrors<uintll_t, 8, computeHamming08,false>(counts);
    else if(n==16)
      countHammingUndetectableErrors<uintll_t, 16, computeHamming16,false>(counts);
    else
      countHammingUndetectableErrors<uintll_t, 24, computeHamming24,false>(counts);
  }

  results_cpu.stop(i_totaltime);
  process_result_hamming(counts, stats, n, h, file_output?"hamming_cpu_native_short_ordering":nullptr);
  delete[] counts;
}
