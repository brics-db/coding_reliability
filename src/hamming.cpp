#include "globals.h"
#include "algorithms.h"
#include <helper.h>

#include <omp.h>
#include <math.h>
#include <iostream>
#include <string.h>
using namespace std;

template<uintll_t N>
inline uintll_t apply_w(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (bitcount(value & 0x56AAAD5B) & 0x1);
  hamming |= (bitcount(value & 0x9B33366D) & 0x1) << 1;
  hamming |= (bitcount(value & 0xE3C3C78E) & 0x1) << 2;
  hamming |= (bitcount(value & 0x03FC07F0) & 0x1) << 3;
 if(N<16)
  return (value << 4) | hamming;
 else // >=16
  hamming |= (bitcount(value & 0x03FFF800) & 0x1) << 4;
 if(N<32)
  return (value << 5) | hamming;
 else // >=32
  hamming |= (bitcount(value & 0xFC000000) & 0x1) << 5;
  return (value << 6) | hamming;
}

template<uintll_t N>
void hamming_cpu(uint128_t* counts, int with_1bit)
{
  const uintll_t end = 1ull<<N;
  const uintll_t h = ( N==8 ? 5 : (N<32?6:7) );
  const uintll_t count_counts = N+h+1;


  #pragma omp parallel
  {
    uintll_t tcounts[count_counts] = {0};
    uint_t d;
/*    #pragma omp master
      printf("Using %d threads\n",omp_get_num_threads());
*/
    #pragma omp for schedule(static)
    for(uintll_t a=1; a<end; ++a)
    {
      d = bitcount(apply_w<N>(a));
      ++tcounts[d];
    }

    for(uintll_t d=2; d<count_counts; ++d) {
      #pragma omp atomic
      counts[d] += static_cast<uint128_t>(tcounts[d]);
    }

  }
  counts[0] = 1ull<<N;
  counts[1] = (N+h)*counts[0];
  //counts[2] = 0;
  for(uint_t d=4; d<count_counts; d+=2)
  {
    counts[d] = counts[d] + counts[d-1];
    counts[d-1] = 0;
  }
  if(with_1bit)
  {
  // 1-Bit spheres, which are detected by ExtHamming
    for(uint_t d=3; d<count_counts; d+=2)
    {
      if(d<N+h){
        counts[d] = static_cast<uint128_t>(N+h-d+1)*counts[d-1] + static_cast<uint128_t>(d+1)*counts[d+1];
      }else
        counts[d] = static_cast<uint128_t>(N+h-d+1)*counts[d-1];
    }
  }
  for(uint_t d=3; d<count_counts; ++d){
    counts[d] = counts[d]<<(static_cast<uint128_t>(N));
  }
}
void run_hamming_cpu(uintll_t n, int with_1bit, int file_output)
{
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);

  const uintll_t h = ( n==8 ? 5 : (n<32?6:7) );
  uint128_t* counts = new uint128_t[n+h+1];
  memset(counts, 0, (n+h+1)*sizeof(uint128_t));

  printf("Start Hamming Coding Algorithm (CPU)\n");
  results_cpu.start(i_totaltime);

  if(n==8)
    hamming_cpu<8>(counts, with_1bit);
  else if(n==16)
    hamming_cpu<16>(counts, with_1bit);
  else if(n==24)
    hamming_cpu<24>(counts, with_1bit);
  else if(n==32)
    hamming_cpu<32>(counts, with_1bit);
  else if(n==40)
    hamming_cpu<40>(counts, with_1bit);

  results_cpu.stop(i_totaltime);

  // h+1 because we removed last parity flag for optimizing counting algorithm
  process_result_hamming(counts, stats, n, h, file_output?"hamming_cpu":nullptr);

  delete[] counts;

}
