#include "globals.h"
#include "algorithms.h"
#include <helper.h>

#include <omp.h>
#include <math.h>
#include <iostream>
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
         uintll_t SZ_SHARDS = 64ull, 
         computeHamming_ft func, 
         uintll_t BITCNT_HAMMING = (BITCNT_DATA == 8 ? 5 : ((BITCNT_DATA == 16) | (BITCNT_DATA == 24) ? 6 : 7)), 
         uintll_t BITCNT_MSG = BITCNT_DATA + BITCNT_HAMMING, 
         uintll_t CNT_COUNTS = BITCNT_MSG + 1ull, 
         uintll_t CNT_EDGES_SHIFT = BITCNT_DATA + BITCNT_MSG, 
         uintll_t CNT_EDGES = 0x1ull << CNT_EDGES_SHIFT, 
         uintll_t CNT_MESSAGES = 0x1ull << BITCNT_DATA, 
         uintll_t MUL_1DISTANCE = BITCNT_MSG,
         uintll_t MUL_2DISTANCE = BITCNT_MSG * (BITCNT_MSG - 1ull) / 2ull, 
         uintll_t CNT_SLICES = CNT_MESSAGES / SZ_SHARDS, 
         uintll_t CNT_SHARDS = CNT_SLICES * CNT_SLICES>
void countHammingUndetectableErrors(uint128_t* result_counts, int with_1bit) 
{
  uintll_t counts[CNT_COUNTS] = { 0 };
  double shardsDone = 0.0;

#pragma omp parallel
  {
/*
#pragma omp master
    {
      cout << "OpenMP using " << omp_get_num_threads() << " threads" << endl;
    }*/
#pragma omp for schedule(dynamic,1)
#ifdef _MSC_VER
    for (__int64 shardX = 0; shardX < CNT_MESSAGES; shardX += SZ_SHARDS) {
#else
      for (uintll_t shardX = 0; shardX < CNT_MESSAGES; shardX += SZ_SHARDS) {
#endif
        uintll_t counts_local[CNT_COUNTS] = { 0 };
        T messages[SZ_SHARDS] = { 0 };
        T messages2[SZ_SHARDS] = { 0 };
        uintll_t distance;

        // 1) precompute Hamming values for this slice, i.e. the "originating" codewords
        for (uintll_t x = shardX, k = 0; k < SZ_SHARDS; ++x, ++k) {
          messages[k] = func(x);
        }

        // 2) Triangle for main diagonale
        for (uintll_t x = 0; x < SZ_SHARDS; ++x) {
          distance = computeDistance(messages[x], messages[x]);
          ++counts_local[distance];
          for (uintll_t y = (x + 1); y < SZ_SHARDS; ++y) {
            distance = computeDistance(messages[x], messages[y]);
            counts_local[distance] += 2;
          }
        }

        // 3) Remainder of the slice
        for (uintll_t shardY = shardX + SZ_SHARDS; shardY < CNT_MESSAGES; shardY += SZ_SHARDS) {
          // 3.1) Precompute other code words
          for (uintll_t y = shardY, k = 0; k < SZ_SHARDS; ++y, ++k) {
            messages2[k] = func(y);
          }

          // 3.2) Do the real work
          for (uintll_t x = 0; x < SZ_SHARDS; ++x) {
            for (uintll_t y = 0; y < SZ_SHARDS; ++y) {
              distance = computeDistance(messages[x], messages2[y]);
              counts_local[distance] += 2;
            }
          }
        }

        // 4) Sum the counts
        for (uintll_t i = 0; i < CNT_COUNTS; ++i) {
#pragma omp atomic
          counts[i] += counts_local[i];
        }

        uintll_t shardsComputed = CNT_SLICES - (static_cast<float>(shardX) / SZ_SHARDS);
        float inc = static_cast<float>(shardsComputed * 2 - 1) / CNT_SHARDS * 100;

#pragma omp atomic
        shardsDone += inc;

/*        if (omp_get_thread_num() == 0) {
          cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << flush;*/
        
      }
    }
  
    if(with_1bit)
    {    
/*    cout << "\b\b\b\b\b\b\b\b\b\b" << right << setw(9) << setprecision(5) << shardsDone << '%' << flush;*/

      counts[1] = counts[0] * BITCNT_MSG;
      cout << dec << "\n#Distances:\n";
      for (uintll_t i = 3; i < CNT_COUNTS; i += 2) {
        counts[i] = (counts[i - 1] * (BITCNT_MSG - i + 1)) + ((i + 1) < CNT_COUNTS ? (counts[i + 1] * (i + 1)) : 0);
      }
      for (uintll_t i = 0; i < CNT_COUNTS; ++i) {
        cout << "  " << right << setw(2) << i << ": " << setw(13) << counts[i] << '\n';
      }
    }
    //cout << "Computation took " << sw << "ns." << endl;

  for(uintll_t i=0;i<CNT_COUNTS; ++i)
    result_counts[i] = static_cast<uint128_t>(counts[i]);
}

void run_hamming_cpu_native(uintll_t n, int with_1bit, int file_output)
{
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  
  const uintll_t h = ( n==8 ? 5 : (n<32?6:7) );
  uint128_t* counts = new uint128_t[n+h+1];

  printf("Start Hamming Coding Algorithm - Native Approach (CPU)\n");
  results_cpu.start(i_totaltime);

  if(n==8)
    countHammingUndetectableErrors<uintll_t, 8, 8, computeHamming08>(counts, with_1bit);
  else if(n==16)
    countHammingUndetectableErrors<uintll_t, 16, 512, computeHamming16>(counts, with_1bit);
  else
    countHammingUndetectableErrors<uintll_t, 24, 512, computeHamming24>(counts, with_1bit);

//    countHammingUndetectableErrors<uint64_t, 32>();

  results_cpu.stop(i_totaltime);
  process_result_hamming(counts, stats, n, h, file_output?"hamming_cpu_native":nullptr);
  delete[] counts;
}
