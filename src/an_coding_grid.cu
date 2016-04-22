
#include "globals.h"
#include "algorithms.h"
#include "an_coding.h"
#include <helper.h>

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>
#include <ostream>


template<uintll_t ShardSize,uint_t CountCounts>
__global__
void dancoding_grid_1d(uintll_t n, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uintll_t iterations, double stepsize)
{
  uintll_t shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid>=end)
    return;

  uintll_t counts_local[CountCounts] = { 0 };

  uintll_t w = A * shardXid * ShardSize;
  uintll_t wend = A * (shardXid+1) * ShardSize;
  uintll_t v;
  uintll_t it = 0;

  for(;w<wend;w+=A)
  {
    for(it=0; it<iterations; ++it)
    {
      v = it*stepsize;
      v *= A;
      ++counts_local[ __popcll( w^v ) ];
    }
  }
  for(int c=0; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}

template<uintll_t ShardSize,uint_t CountCounts>
__global__
void dancoding_grid_2d(uintll_t n, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uintll_t iterations, double stepsize)
{
  uintll_t shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid>=end)
    return;

  uintll_t counts_local[CountCounts] = { 0 };
  uintll_t s = stepsize;
  uintll_t w = A * shardXid * ShardSize * s;
  uintll_t w_end = A * (shardXid+1) * ShardSize * s;
  uintll_t v;
  uintll_t it = 0;

  for( ; w<w_end; w += A*s)
  {
    for(it=0; it<iterations; ++it)
    {
      v = it*stepsize;
      v *= A;
      ++counts_local[ __popcll( w^v ) ];
    }    
  }
  for(int c=0; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}

/**
 * Caller for kernel 
 */
template<uintll_t N>
struct Caller1D
{
  template<typename T>
  void operator()(dim3 blocks, dim3 threads, uintll_t n, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uintll_t iterations, T stepsize){
    dancoding_grid_1d<ANCoding::traits::Shards<N>::value, ANCoding::traits::CountCounts<N>::value ><<< blocks, threads >>>(n, A, counts, offset, end, iterations, stepsize);
  }
};
template<uintll_t N>
struct Caller2D
{
  template<typename T>
  void operator()(dim3 blocks, dim3 threads, uintll_t n, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uintll_t iterations, T stepsize){
    dancoding_grid_2d<ANCoding::traits::Shards<N>::value, ANCoding::traits::CountCounts<N>::value ><<< blocks, threads >>>(n, A, counts, offset, end, iterations, stepsize);
  }
};

double run_ancoding_grid(int gdim, uintll_t n, uintll_t iterations, uintll_t A, int verbose, double* times, uintll_t* minb, uintll_t* mincb, int file_output, int nr_dev_max)
{
  assert(A<(1ull<<n));
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
  if(verbose){
    printf("Start AN-Coding Algorithm - %dD Grid with %zu points\n", gdim, iterations);
    printf("Found %d CUDA devices (%s).\n", nr_dev, prop.name);
  }
  // skip init time
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  uint_t n_up = n<=8 ? 8 : n<=16 ? 16 : n<=32 ? 32 : n<=40 ? 40 : 48;
  const uintll_t count_messages = (1ull << n);
  const uintll_t size_shards = ANCoding::getShardsSize(n_up); 
  iterations = iterations>count_messages?count_messages:iterations;

  const uintll_t count_shards = (gdim==2 ? iterations : count_messages) / size_shards;
  const uintll_t bitcount_A = ceil(log((double)A)/log(2.0));
  const uint_t count_counts = n + bitcount_A + 1;

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

    if(gdim==1)
      ANCoding::bridge<Caller1D>(n_up, blocks, threads, n, A, dcounts[dev], offset, end, iterations, 1.0L*count_messages/iterations);
    else
      ANCoding::bridge<Caller2D>(n_up, blocks, threads, n, A, dcounts[dev], offset, end, iterations, 1.0L*count_messages/iterations);

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

  if(verbose)
  {
    process_result_ancoding_mc(counts,stats,n,A,iterations,file_output?"ancoding_grid1d":nullptr);
  }

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }


  return max_abs_error;
}
