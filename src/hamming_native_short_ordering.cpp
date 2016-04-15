#include "globals.h"
#include "algorithms.h"
#include "hamming.h"
#include "helper.h"

#include <omp.h>
#include <math.h>
#include <iostream>
#include <string.h>
using namespace std;

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
