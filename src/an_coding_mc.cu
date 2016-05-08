
#include "globals.h"
#include "an_coding.h"
#include "algorithms.h"
#include "rand_gen.cuh"

#include <helper.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>
#include <ostream>

#include <curand.h>
#include <curand_kernel.h>

__device__ int popc(uintll_t v){ return __popcll(v); }
__device__ int popc(uint_t v){ return __popc(v); }

template<uintll_t ShardSize,uint_t CountCounts, typename T, typename RandGenType>
__global__
void dancoding_mc(T n, T A, uintll_t* counts, T offset, T end, RandGenType* state, T iterations, double p2n)
{
  uint_t tid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
  T shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid>=end)
    return;

  T counts_local[CountCounts] = { 0 };

  T w = A * shardXid * ShardSize;
  T wend = A * (shardXid+1) * ShardSize;
  T v;
  T it;
  RandGenType local_state = state[tid];
  for(;w<wend;w+=A)
  {
    for(it=0; it<iterations; ++it)
    {
      v = static_cast<T>( p2n * curand_uniform_double(&local_state));
      v *= A;
      ++counts_local[ popc( w^v ) ];
    }
  }
  for(int c=0; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
  state[tid] = local_state;
}
/**
 * Caller for kernel 
 */
template<uintll_t N>
struct Caller
{
  template<typename RandGenType>
  void operator()(uintll_t n, dim3 blocks, dim3 threads, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, RandGenType* states, uintll_t iterations){
    double p2n = pow(2.0,n);
    if((A<<n)<(1ull<<32))
      dancoding_mc<ANCoding::traits::Shards<N>::value, ANCoding::traits::CountCounts<N>::value >
        <<< blocks, threads >>>((uint_t)n, (uint_t)A, counts, (uint_t)offset, (uint_t)end, states, (uint_t)iterations, p2n);
   else
      dancoding_mc<ANCoding::traits::Shards<N>::value, ANCoding::traits::CountCounts<N>::value >
        <<< blocks, threads >>>(n, A, counts, offset, end, states, iterations, p2n);
  }
};

double run_ancoding_mc(uintll_t n, uintll_t iterations, uintll_t A, int verbose, double* times, uintll_t* minb, uintll_t* mincb, int file_output, int nr_dev_max)
{
  int tmp_nr_dev;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);


  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = nr_dev_max==0 ? tmp_nr_dev : min(nr_dev_max,tmp_nr_dev);
  cudaDeviceProp prop;
  CHECK_ERROR( cudaGetDeviceProperties(&prop, 0));
  if(verbose>1){
    printf("Start AN-Coding Algorithm - Monte Carlo with %zu iterations\n", iterations);
    printf("Found %d CUDA devices (%s).\n", nr_dev, prop.name);
  }
  // skip init time
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaThreadSetCacheConfig(cudaFuncCachePreferL1) );
    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  const uintll_t size_shards = ANCoding::getShardsSize(n);
  const uintll_t count_shards = count_messages / size_shards;
  const uintll_t bitcount_A = ceil(log((double)A)/log(2.0));
  const uint_t count_counts = n + bitcount_A + 1;

  uintll_t** dcounts;
  uintll_t** hcounts;

  dcounts = new uintll_t*[nr_dev];
  hcounts = new uintll_t*[nr_dev];
  iterations = iterations>count_messages?count_messages:iterations;

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    uint_t xblocks;
    uintll_t offset, end;
    dim3 blocks;
    RandGen<RAND_GEN> gen;

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
    if(verbose>1){
      printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
      printf("Dev %d: Blocks: %d %d, offset %llu, end %llu, end %llu\n", dev, blocks.x, blocks.y, offset, end, (threads.x-1+threads.x * ((xblocks-1) * (xblocks) + (xblocks-1)) + offset)*size_shards);
    }

    /* random generator stuff */
    gen.init(blocks, threads, 1337+8137*xblocks*xblocks*threads.x*dev, 1, dev);
    //dim3 blocks( (count_shards / threads.x)/2, 2 );
    if(dev==0)
      results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, threads, A, dcounts[dev], offset, end, gen.devStates, iterations);

    CHECK_LAST("Kernel failed.");

    if(dev==0) results_gpu.stop(i_runtime);

    gen.free();
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
//  counts[0] = 1ull<<n;
  for(uint_t i=0; i<count_counts; ++i)
  {
    counts[i] = static_cast<uint128_t>(static_cast<long double>(pow(2.0,n)/iterations*hcounts[0][i]));
    //<<1;//only <<1 if sorted
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;
  CHECK_ERROR( cudaDeviceReset() );

  // compute max. relative error
  double max_abs_error = get_rel_error_AN(A, n, counts, 0);

  if(minb!=nullptr && mincb!=nullptr)
  {
    *minb=0xFFFF;;
    *mincb=static_cast<uintll_t>(-1);
    for(uint_t i=1; i<count_counts/2; ++i)
    {
      if(counts[i]!=0 && counts[i]<static_cast<uint128_t>(*mincb))
      {
        *minb=i;
        *mincb=counts[i];
      }
    }
  }

  if(verbose || file_output)
  {
    if(nr_dev==4)
      process_result_ancoding_mc(counts,stats,n,A,iterations,file_output?"ancoding_mc_4gpu":nullptr);
    else
      process_result_ancoding_mc(counts,stats,n,A,iterations,file_output?"ancoding_mc_gpu":nullptr);
  }

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }


  return max_abs_error;
}
