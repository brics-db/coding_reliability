// Copyright 2016 Matthias Werner
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

template<uint_t CountCounts, typename T, typename TReal>
__global__
void dancoding_grid_1d(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize)
{
  T counts_local[CountCounts] = { 0 };
  T v, w, k;
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset; 
       i < end; 
       i += blockDim.x * gridDim.x) 
  {
    w = A*i;
    for(v=0, k=0; v<Aend; ++k, v=A*static_cast<T>(k*stepsize))
    {
      ++counts_local[ ANCoding::dbitcount( w^v ) ];
    }
  }
  for(uint_t c=1; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}

template<uint_t CountCounts, typename T, typename TReal>
__global__
void dancoding_grid_2d(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize, TReal stepsize2)
{

  T counts_local[CountCounts] = { 0 };
  T v, w, k;
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset; 
       i < end; 
       i += blockDim.x * gridDim.x) 
  {
    w = A*static_cast<T>(i*stepsize2);
    for(v=0, k=0; v<Aend; ++k, v=A*static_cast<T>(k*stepsize))
    {
      ++counts_local[ ANCoding::dbitcount( w^v ) ];
    }
  }
  for(int c=1; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}


/**
 * Caller for kernel 
 */
template<uintll_t N>
struct Caller
{
  template<typename T>
  void operator()(uintll_t n, dim3 blocks, dim3 threads, int gdim, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, T stepsize, T stepsize2){
    uintll_t Aend = A<<n;
    bool use32bit = Aend<(1ull<<32);
    if(gdim==1){
      if(use32bit)
        dancoding_grid_1d<ANCoding::traits::CountCounts<N>::value, uint_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize);
      else   
        dancoding_grid_1d<ANCoding::traits::CountCounts<N>::value, uintll_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize);
    }else{
      if(use32bit)
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uint_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
      else
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uintll_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
    }
  }
};


double run_ancoding_grid(int gdim, uintll_t n, uintll_t iterations, uintll_t iterations2, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb, int file_output, int nr_dev_max)
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
  if(verbose>1){
    printf("Start AN-Coding Algorithm - %dD Grid with %llu x %llu points\n", gdim, iterations, gdim==2?iterations2:1);
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
  iterations = iterations>count_messages?count_messages:iterations;
  iterations2= gdim==1 ? count_messages : iterations2>count_messages ? count_messages : iterations2;

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
    uintll_t offset, end;
    dim3 blocks;

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
    CHECK_ERROR( cudaMalloc(dcounts+dev, count_counts*sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts*sizeof(uintll_t)) );

    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );
    
    hcounts[dev] = new uintll_t[count_counts];

    //end = 0;
    //    offset = count_shards / nr_dev / nr_dev * (dev)*(dev);
    //    end = count_shards / nr_dev / nr_dev * (dev+1)*(dev+1);
    offset = iterations2 / nr_dev * dev;
    end = iterations2 / nr_dev * (dev+1);

    blocks.x = 8*numSMs;

    // 3) Remainder of the slice
    if(verbose>1){
      printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d %d, offset %llu, end %llu, end %llu\n", dev, numSMs, blocks.x, blocks.y, offset, end, (threads.x-1+threads.x * ((blocks.x-1) * (blocks.x) + (blocks.x-1)) + offset));
    }
    if(dev==0)
      results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, threads, gdim, A, dcounts[dev], offset, end, 1.0*count_messages/iterations, 1.0*count_messages/iterations2);

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
  counts[0] = 1ull<<n;
  long double factor = gdim==1 ? pow(2.0L,n)/iterations : pow(4.0L,n)/(iterations*iterations2);
  for(uint_t i=0; i<count_counts; ++i)
  {
    counts[i] = static_cast<uint128_t>(factor*hcounts[0][i]);
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
    *minb=0xFFFF;
    *mincb=static_cast<uint128_t>(-1);
    for(uint_t i=1; i<count_counts/2; ++i)
    {
      if(counts[i]!=0 && counts[i]<*mincb)
      {
        *minb=i;
        *mincb=counts[i];
        break;
      }
    }
  }

  if(verbose || file_output)
  {
    char fname[256];
    sprintf(fname, "ancoding_grid%dd_%s", gdim, nr_dev==4 ? "4gpu" : "gpu");
    process_result_ancoding_mc(counts,stats,n,A,iterations,file_output ? fname : nullptr);
  }

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }


  return max_abs_error;
}
