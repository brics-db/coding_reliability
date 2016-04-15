#include "globals.h"
#include "algorithms.h"

#include <helper.h>

#include <string.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <omp.h>
#include <list>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcountll __popcnt64
#endif

#define bitcount __builtin_popcountll

/*
 * Count undetectable bit flips for AN encoded data words.
 *
 * AN coding cannot detect bit flips which result in multiples of A.
 * 
 * We essentially only count all transitions from codewords to other possible codewords, i.e. multiples of A.
 * All actual bit flips (i.e. distance > 0) are undetectable bit flips.
 * 
 * Therefore, we model the space of possible messages as a graph. The edges are the transitions / bit flips between words,
 * while the weights are the number of bits required to flip to get to the other message.
 * Since we start from codewords only, the graph is directed, from all codewords to all other codewords an all words in the
 * corresponding 1-distance sphears.
 * 
 *
 * The weights (W) are counted, grouped by the weight. This gives the corresponding number of undetectable W-bitflips.
 */

template<typename T>
inline uintll_t computeDistance(const T &value1, const T &value2) {
  return static_cast<uintll_t>(bitcount(value1 ^ value2));
}

template<uintll_t SZ_SHARDS = 64>
void countANCodingUndetectableErrors(uintll_t n, uintll_t A, uint128_t* counts, uintll_t count_counts) 
{
  double shardsDone = 0.0;

#pragma omp parallel 
  {
    const uintll_t CNT_MESSAGES = 0x1ull << n; 
    const uintll_t CNT_SLICES = CNT_MESSAGES / SZ_SHARDS; 
    const uintll_t CNT_SHARDS = CNT_SLICES * CNT_SLICES;
    uintll_t* counts_local = new uintll_t[count_counts];
    memset(counts_local, 0, count_counts*sizeof(uintll_t));
#pragma omp for schedule(dynamic,1)
    for (uintll_t shardX = 0; shardX < CNT_MESSAGES; shardX += SZ_SHARDS) 
    {
      // 1) Triangle for main diagonale
      uintll_t m1, m2;

      for (uintll_t x = 0; x < SZ_SHARDS; ++x) {
        m1 = (shardX + x) * A;
        ++counts_local[computeDistance(m1, m1)];
        for (uintll_t y = (x + 1); y < SZ_SHARDS; ++y) {
          m2 = (shardX + y) * A;
          counts_local[computeDistance(m1, m2)]+=2;
        }
      }

      // 2) Remainder of the slice
      for (uintll_t shardY = shardX + SZ_SHARDS; shardY < CNT_MESSAGES; shardY += SZ_SHARDS) {
        for (uintll_t x = 0; x < SZ_SHARDS; ++x) {
          m1 = (shardX + x) * A;
          for (uintll_t y = 0; y < SZ_SHARDS; ++y) {
            m2 = (shardY + y) * A;
            counts_local[computeDistance(m1, m2)]+=2;
          }
        }
      }

      uintll_t shardsComputed = CNT_SLICES - (shardX / SZ_SHARDS);
      float inc = static_cast<double>(shardsComputed * 2 - 1) / CNT_SHARDS * 100;
#pragma omp atomic
      shardsDone += inc;
    } // for

    // 3) Sum the counts
    for (uintll_t i = 0; i < count_counts; ++i) {
#pragma omp atomic
      counts[i] += counts_local[i];
    }

    delete[] counts_local;
  } // parallel
}

uintll_t msb(uintll_t v)
{
  uintll_t r = 0;
  while(v>>=1)
    ++r;
  return r;
}


void run_ancoding_cpu(uintll_t n, uintll_t A, int verbose, uintll_t* minb, uintll_t* mincb, int file_output)
{
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);

  const uintll_t count_counts = n+ceil(log((double)A)/log(2.0))+1; 
  uint128_t* counts = new uint128_t[count_counts];
  memset(counts, 0, count_counts * sizeof(uintll_t));

  results_cpu.start(i_totaltime);
   countANCodingUndetectableErrors(n, A, counts, count_counts);
  results_cpu.stop(i_totaltime);
  if(verbose || file_output)
    process_result_ancoding(counts, stats, n, A, file_output?"ancoding_cpu":nullptr);

  delete[] counts; 

}

