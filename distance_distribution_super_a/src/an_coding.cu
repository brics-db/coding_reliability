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
#include <type_traits>

#include <sstream>
#include <iostream>
#include <ostream>
using namespace std;


template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll>
__global__
void dancoding_shared(T n, T A, uintll_t* counts, T offset, T end, T Aend)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ uint_t counts_shared[SharedCount];
  T Aend_p = Aend-Unroll*A;
  for(uint_t i = threadIdx.x; i < SharedCount; i+=SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  T v, w, i;
  uint_t max = 0;
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end;
       i += BlockSize * gridDim.x)
  {
    w = A*i;
    for(v=w+A; v<Aend_p; v=v+Unroll*A)
    {
      #pragma unroll (Unroll)
      for(T u=0; u<Unroll; ++u) {
        uint_t z = ANCoding::dbitcount( w^(v+u*A) )*SharedRowPitch + threadIdx.x;
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
    for(; v<Aend; v+=A)
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

template<uint_t CountCounts, typename T>
__global__
void dancoding(T n, T A, uintll_t* counts, T offset, T end, T Aend)
{
  T counts_local[CountCounts] = { 0 };
  T v, w;
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < end;
       i += blockDim.x * gridDim.x)
  {
    w = A*i;
    for(v=w+A; v<Aend; v+=A)
    {
      ++counts_local[ ANCoding::dbitcount( w^v ) ];
    }
  }
  for(uint_t c=1; c<CountCounts; ++c)
    atomicAdd(counts+c, counts_local[c]);
}

/**
 * Caller for kernel
 * @tparam N N is either 8,16,24,32,40 or 48
 */
template<uintll_t N>
struct Caller
{
  void operator()(uintll_t n, dim3 blocks, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t h){
    static constexpr uint_t blocksize = 128;
    static constexpr uint_t unroll_factor = 16;
    uintll_t Aend = A<<n;
    uint_t ccmax = n+h+1;
    CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    CHECK_ERROR( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );
    if(ccmax<=16)
      dancoding_shared<blocksize, 16, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<=24)
      dancoding_shared<blocksize, 24, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<=32)
      dancoding_shared<blocksize, 32, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<=48)
      dancoding_shared<blocksize, 48, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<=64)
      dancoding_shared<blocksize, 64, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else
      throw std::runtime_error("Not supported");

  }
};


void run_ancoding(uintll_t A, uint_t h, Flags& flags, double* times, uint128_t* counts)
{
  if( ( A & (A-1) ) == 0 ) // A is power of two
    A=1;

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

  uintll_t** dcounts;
  uintll_t** hcounts;
  dcounts = new uintll_t*[nr_dev];
  hcounts = new uintll_t*[nr_dev];

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev=0; dev<nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1); // not used for kernel launch, see Caller
    dim3 blocks(1);
    uintll_t offset, end;
    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(dcounts+dev, count_counts*sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts*sizeof(uintll_t)) );

    hcounts[dev] = new uintll_t[count_counts];
    if(nr_dev>1) {
      // load balancer
      double faca = 1.0 - sqrt( 1.0 - static_cast<double>(dev)/nr_dev );
      double facb = 1.0 - sqrt( 1.0 - static_cast<double>(dev+1)/nr_dev );
      offset = ceil(count_messages * faca);
      end    = ceil(count_messages * facb);
    }else{
      offset = 0;
      end = count_messages;
    }
    // grid-stride
    // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    blocks.x = 128*numSMs;

    if(dev==0)
      results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, A, dcounts[dev], offset, end, h);
    CHECK_LAST("Kernel failed.");

    if(dev==0)
      results_gpu.stop(i_runtime);

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
  for(uint_t i=1; i<count_counts; ++i)
  {
    counts[i] = hcounts[0][i]<<1;
  }

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;

}
