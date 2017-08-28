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
#include <sstream>
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

template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll, typename TReal>
__global__
void dancoding_grid_1d_shared(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ uint_t counts_shared[SharedCount];

  T Aend_p = Aend-A*static_cast<T>(Unroll*stepsize);
  for(uint_t i = threadIdx.x; i < SharedCount; i+=SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  T v, w, i, k;
  uint_t max = 0;
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end; 
       i += BlockSize * gridDim.x)
  {
    w = A*i;
    for(v=0, k=0; v<Aend_p; k+=Unroll)
    {
      #pragma unroll (Unroll)
      for(T u=0; u<Unroll; ++u, v=A*static_cast<T>((k+u)*stepsize))
      {
        uint_t z = ANCoding::dbitcount( w^v )*SharedRowPitch + threadIdx.x;
        ++counts_shared[ z ];
        if(counts_shared[z]>max) max = counts_shared[z];
      }
      if(max > ((1u<<31)-Unroll)<<1) { // prevent overflows by flushing intermediate results (2^32 - 2*Unroll)
        for(uint_t c=0; c<CountCounts; ++c) {
          atomicAdd(&counts[c], counts_shared[c*SharedRowPitch+threadIdx.x]);
          max = 0;
          counts_shared[c*SharedRowPitch+threadIdx.x] = 0;
        }
      }
    }
    for(; v<Aend; ++k, v=A*static_cast<T>(k*stepsize))
    {
      uint_t z = ANCoding::dbitcount( w^v )*SharedRowPitch + threadIdx.x;
      ++counts_shared[ z ];
    }
  }
  __syncthreads();
  if(threadIdx.x<CountCounts) {
    for(uint_t c=0; c<BlockSize; ++c)
      atomicAdd(&counts[threadIdx.x], static_cast<uintll_t>(counts_shared[threadIdx.x*SharedRowPitch+c]));
  }
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
  void operator()(uintll_t n, dim3 blocks, dim3 threads, int gdim, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t ccmax, T stepsize, T stepsize2, uint_t iterations){
    static constexpr uint_t blocksize = 128;
    static constexpr uint_t unroll_factor = 16;
    uintll_t Aend = A<<n;

    if(gdim==1){
      assert(iterations>=3);
      CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
      CHECK_ERROR( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );

      if(iterations<=16) {
        if(ccmax<=16)
          dancoding_grid_1d_shared<blocksize, 16, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=24)
          dancoding_grid_1d_shared<blocksize, 24, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=32)
          dancoding_grid_1d_shared<blocksize, 32, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=48)
          dancoding_grid_1d_shared<blocksize, 48, uintll_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=64)
          dancoding_grid_1d_shared<blocksize, 64, uintll_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else
          throw std::runtime_error("Not supported");
      } else {
        if(ccmax<=16)
          dancoding_grid_1d_shared<blocksize, 16, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=24)
          dancoding_grid_1d_shared<blocksize, 24, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=32)
          dancoding_grid_1d_shared<blocksize, 32, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=48)
          dancoding_grid_1d_shared<blocksize, 48, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax<=64)
          dancoding_grid_1d_shared<blocksize, 64, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else
          throw std::runtime_error("Not supported");
      }

    }else{
      if(Aend<(1ull<<32))
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uint_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
      else
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uintll_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
    }
  }
};


void run_ancoding_grid(uintll_t A, uint_t h, Flags& flags, double* times, uint128_t* counts)
{
  if( ( A & (A-1) ) == 0 ) // A is power of two
    A=1;
  assert(A<(1ull<<flags.n));

  int tmp_nr_dev;
  Statistics stats;
  TimeStatistics results_cpu (&stats,CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats,GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);

  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = flags.nr_dev==0 ? tmp_nr_dev : min(flags.nr_dev,tmp_nr_dev);

  results_cpu.start(i_totaltime);

  uint_t n = flags.n;
  const uintll_t count_messages = (1ull << n);
  const uint_t count_counts = n + h + 1;

  uintll_t iterations = flags.mc_iterations > count_messages ? count_messages : flags.mc_iterations;
  uintll_t iterations2 = flags.with_grid==1 ? count_messages : flags.mc_iterations_2>count_messages ? count_messages : flags.mc_iterations_2;

  uintll_t** dcounts;
  uintll_t** hcounts;

  dcounts = new uintll_t*[nr_dev];
  hcounts = new uintll_t*[nr_dev];

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    dim3 blocks;
    uintll_t offset, end;
    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(dcounts+dev, count_counts*sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts*sizeof(uintll_t)) );


    hcounts[dev] = new uintll_t[count_counts];

    //end = 0;
    //    offset = count_shards / nr_dev / nr_dev * (dev)*(dev);
    //    end = count_shards / nr_dev / nr_dev * (dev+1)*(dev+1);
    offset = iterations2 / nr_dev * dev;
    end = iterations2 / nr_dev * (dev+1);

    blocks.x = 128*numSMs;

    // 3) Remainder of the slice
    if(dev==0)
      results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, threads, flags.with_grid, A, dcounts[dev], offset, end, count_counts, 1.0*count_messages/iterations, 1.0*count_messages/iterations2, iterations);

    CHECK_LAST("Kernel failed.");

    if(dev==0) results_gpu.stop(i_runtime);

  }

  CHECK_ERROR(cudaMemcpy(hcounts[0], dcounts[0], count_counts*sizeof(uintll_t), cudaMemcpyDefault));

  // other devices sum up to [0]
  for(int dev=1; dev<nr_dev; ++dev)
  {
    CHECK_ERROR(cudaMemcpy(hcounts[dev], dcounts[dev], count_counts*sizeof(uintll_t), cudaMemcpyDefault));
    for(uint_t i=0; i<count_counts; ++i)
      hcounts[0][i] += hcounts[dev][i];
    CHECK_ERROR( cudaFree(dcounts[dev]) );
    delete[] hcounts[dev];
  }

  results_cpu.stop(i_totaltime);

  // results
  counts[0] = 1ull<<n;
  long double factor = flags.with_grid==1 ? pow(2.0L,n)/iterations : pow(4.0L,n)/(iterations*iterations2);
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

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }

}
