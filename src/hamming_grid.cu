
#include "globals.h"
#include "algorithms.h"
#include "hamming.h"
#include <helper.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>
#include <ostream>
#include <type_traits>

template<uintll_t N>
__device__ inline uintll_t computeHamming(const uintll_t &value) {
  uintll_t hamming = 0;
  hamming |= (__popcll(value & 0x56AAAD5B) & 0x1);
  hamming |= (__popcll(value & 0x9B33366D) & 0x1) << 1;
  hamming |= (__popcll(value & 0xE3C3C78E) & 0x1) << 2;
  hamming |= (__popcll(value & 0x03FC07F0) & 0x1) << 3;
 if(N<16)
  return (value << 4) | hamming;
 else // >=16
  hamming |= (__popcll(value & 0x03FFF800) & 0x1) << 4;
 if(N<32)
  return (value << 5) | hamming;
 else // >=32
  hamming |= (__popcll(value & 0xFC000000) & 0x1) << 5;
  return (value << 6) | hamming;
}

template<uintll_t N, uintll_t ShardSize,int CountCounts>
__global__
void dhamming_grid_1d(uintll_t* counts, uintll_t offset, uintll_t end, double stepsize)
{
  uintll_t shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid>=end)
    return;

  uintll_t counts_local[CountCounts] = { 0 };
  uintll_t w = shardXid * ShardSize * (uintll_t)stepsize;
  uintll_t x;
  
  for(uintll_t k=0; k<ShardSize; ++k)
  {
    x = w;
    x += k*stepsize;
    ++counts_local[ __popcll( computeHamming<N>( x ) ) ];
  }

  for(int c=0; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}

/**
 * Caller for kernel 
 */
template<uintll_t N>
struct Caller
{
  template<typename T>
  void operator()(dim3 blocks, dim3 threads, uintll_t* counts, uintll_t offset, uintll_t end, T stepsize){
    dhamming_grid_1d<N, Hamming::traits::Shards<N>::value, Hamming::traits::CountCounts<N>::value ><<< blocks, threads >>>(counts, offset, end, stepsize);
  }
};


double run_hamming_grid(uintll_t n, int with_1bit, uintll_t iterations, int file_output, int nr_dev_max)
{
  int tmp_nr_dev;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);
  const int verbose = 1;

  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = nr_dev_max==0 ? tmp_nr_dev : min(nr_dev_max,tmp_nr_dev);
  cudaDeviceProp prop;
  CHECK_ERROR( cudaGetDeviceProperties(&prop, 0));
  if(verbose){
    printf("Start 1D Hamming-Coding Algorithm - 1D Grid with %zu iterations\n", iterations);
    printf("Found %d CUDA devices (%s).\n", nr_dev, prop.name);
  }
  // skip init time
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  const uintll_t size_shards = Hamming::getShardsSize(n); // also used in template kernel launch
  iterations = iterations>count_messages?count_messages:iterations;

  const uintll_t count_shards = iterations / size_shards; // parallelize over iterations number
  const uint_t h = ( n==8 ? 5 : (n<32?6:7) );
  const uintll_t bitcount_message = n + h;
  const uint_t count_counts = bitcount_message + 1;


  uintll_t** dcounts;
  uintll_t** hcounts;

  dcounts = new uintll_t*[nr_dev];
  hcounts = new uintll_t*[nr_dev];

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    uint_t xblocks;
    uintll_t offset, end;
    dim3 blocks;

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
    CHECK_ERROR( cudaMalloc(dcounts+dev, count_counts*sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts*sizeof(uintll_t)) );

    hcounts[dev] = new uintll_t[count_counts];
    memset(hcounts[dev], 0, count_counts*sizeof(uintll_t));


    //end = 0;
    //    offset = count_shards / nr_dev / nr_dev * (dev)*(dev);
    //    end = count_shards / nr_dev / nr_dev * (dev+1)*(dev+1);
    offset = count_shards / nr_dev * dev;
    end = count_shards / nr_dev * (dev+1);

    xblocks = ceil(sqrt(1.0*(end-offset) / threads.x)) ;
    blocks.x= xblocks; blocks.y = xblocks;

    // 3) Remainder of the slice
    if(verbose){
      printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
      printf("Dev %d: Blocks: %d %d, offset %llu, end %llu, end %llu\n", dev, blocks.x, blocks.y, offset, end, (threads.x-1+threads.x * ((xblocks-1) * (xblocks) + (xblocks-1)) + offset)*size_shards);
    }

    if(dev==0)
      results_gpu.start(i_runtime);

    Hamming::bridge<Caller>(n, blocks, threads, dcounts[dev], offset, end, 1.0L*count_messages/iterations);

    CHECK_LAST("Kernel failed.");

    if(dev==0) results_gpu.stop(i_runtime);

  }

  CHECK_ERROR(
      cudaMemcpy(hcounts[0], dcounts[0], count_counts*sizeof(uintll_t), cudaMemcpyDefault)
      );
  // other devices sum up to [0]
  for(int dev=1; dev<nr_dev; ++dev)
  {
    CHECK_ERROR(
      cudaMemcpy(hcounts[dev], dcounts[dev], count_counts*sizeof(uintll_t), cudaMemcpyDefault)
      );
    for(uint_t i=0; i<count_counts; ++i)
      hcounts[0][i] += hcounts[dev][i];
    CHECK_ERROR( cudaFree(dcounts[dev]) );
    delete[] hcounts[dev];
  }

  results_cpu.stop(i_totaltime);


  // results
  uint128_t counts[64] = {0};
  counts[0] = static_cast<uint128_t>(static_cast<long double>(pow(2.0,n)*(hcounts[0][0])));
  for(uint_t i=2; i<count_counts; i+=2)
  {
    counts[i] = static_cast<uint128_t>(static_cast<long double>(pow(2.0,n)*(hcounts[0][i]+hcounts[0][i-1])));
  }
  if(with_1bit)
  {  
    // 1-bit sphere  
    for (uint_t i = 1; i < count_counts; i+=2)
    {
      if(i+1<count_counts){
        counts[i] = uint128_t(i+1)*counts[i+1] + uint128_t(bitcount_message-i+1)*counts[i-1];
      }else
        counts[i] = uint128_t(bitcount_message-i+1)*counts[i-1];
    }
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;
  CHECK_ERROR( cudaDeviceReset() );
  
  double max_abs_error = get_abs_error_hamming(n, counts, 0, with_1bit, nullptr);

  if(verbose)
  {    
    process_result_hamming_mc(counts,stats,n,h,with_1bit,iterations,file_output?"hamming_mc":nullptr);
  }

  return max_abs_error;
}
