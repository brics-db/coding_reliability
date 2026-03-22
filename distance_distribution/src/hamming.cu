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
#include "hamming.h"

#include <helper.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <iostream>
#include <ostream>
#include <vector>

using namespace std;

/**
 * @brief Device function to compute the Extended Hamming codeword for a given data value.
 * 
 * @tparam N Number of data bits.
 * @param value The input data bits.
 * @return The codeword including parity bits.
 */
template<uintll_t N>
__device__ inline uintll_t computeHamming(const uintll_t &value) {
  uintll_t hamming = 0;
  // Parity bit calculations for various bit positions
  hamming |= (__popcll(value & 0x56AAAD5B) & 0x1);
  hamming |= (__popcll(value & 0x9B33366D) & 0x1) << 1;
  hamming |= (__popcll(value & 0xE3C3C78E) & 0x1) << 2;
  if(N < 8)
    return (value << 3) | hamming;
  else
    hamming |= (__popcll(value & 0x03FC07F0) & 0x1) << 3;
  if(N < 16)
    return (value << 4) | hamming;
  else // >= 16
    hamming |= (__popcll(value & 0x03FFF800) & 0x1) << 4;
  if(N < 32)
    return (value << 5) | hamming;
  else // >= 32
    hamming |= (__popcll(value & 0xFC000000) & 0x1) << 5;
  return (value << 6) | hamming;
}

/**
 * @brief CUDA kernel for computing Hamming distance distribution.
 */
template<uintll_t N, uintll_t ShardSize, int CountCounts>
__global__
void dhamming(uintll_t* counts, uintll_t offset, uintll_t end)
{
  uintll_t shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid >= end)
    return;

  uintll_t counts_local[CountCounts] = { 0 };

  uintll_t w = shardXid * ShardSize;
  for(uintll_t k = 0; k < ShardSize; ++k)
  {
    ++counts_local[ __popcll( computeHamming<N>( w + k ) ) ];
  }
  for(int c = 0; c < CountCounts; ++c)
    atomicAdd(counts + c, counts_local[c]);
}

/**
 * @brief Helper structure to launch the appropriate Hamming kernel.
 */
template<uintll_t N>
struct Caller
{
  void operator()(dim3 blocks, dim3 threads, uintll_t* counts, uintll_t offset, uintll_t end){
    dhamming<N, Hamming::traits::Shards<N>::value, Hamming::traits::CountCounts<N>::value ><<< blocks, threads >>>(counts, offset, end);
  }
};

/**
 * @brief Host function to run the Hamming distance distribution calculation.
 */
void run_hamming(uintll_t n, int with_1bit, int file_output, int nr_dev_max)
{
  Statistics stats;
  TimeStatistics results_cpu (&stats, CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats, GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);
  const uint_t verbose = 1;

  int tmp_nr_dev;
  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = nr_dev_max == 0 ? tmp_nr_dev : min(nr_dev_max, tmp_nr_dev);
  
  if(verbose > 1) {
    cudaDeviceProp prop;
    CHECK_ERROR( cudaGetDeviceProperties(&prop, 0));
    printf("Start Hamming Coding Algorithm\n");
    printf("Found %d CUDA devices (%s).\n", nr_dev, prop.name);
  }

  // Initialize devices
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    // Modernize: Replace deprecated cudaThreadSetCacheConfig
    CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
    CHECK_ERROR( cudaDeviceSynchronize() );
  }
  
  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  const uintll_t size_shards = Hamming::getShardsSize(n);
  const uintll_t count_shards = count_messages / size_shards;
  const uint_t h = n == 4 ? 4 : ( n == 8 ? 5 : (n < 32 ? 6 : 7) );
  const uintll_t bitcount_message = n + h;
  const uint_t count_counts = bitcount_message + 1;
  
  uintll_t** dcounts = new uintll_t*[nr_dev];
  uintll_t** hcounts = new uintll_t*[nr_dev];

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    uintll_t offset, end;

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(&dcounts[dev], count_counts * sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts * sizeof(uintll_t)) );

    hcounts[dev] = new uintll_t[count_counts];
    memset(hcounts[dev], 0, count_counts * sizeof(uintll_t));
  
    offset = count_shards / nr_dev * dev;
    end = count_shards / nr_dev * (dev + 1);

    uint_t xblocks = ceil(sqrt(1.0 * (end - offset) / threads.x));
    dim3 blocks(xblocks, xblocks);

    if(dev == 0 && verbose > 1) {
      printf("Using %d threads, blocks %u, %llu-%llu.\n", omp_get_num_threads(), xblocks, offset, end);
    }

    if(dev == 0) results_gpu.start(i_runtime);

    Hamming::bridge<Caller>(n, blocks, threads, dcounts[dev], offset, end);
    CHECK_LAST("Kernel failed.");

    if(dev == 0) results_gpu.stop(i_runtime);
  }

  // Aggregate results back to host
  CHECK_ERROR( cudaMemcpy(hcounts[0], dcounts[0], count_counts * sizeof(uintll_t), cudaMemcpyDeviceToHost) );
  for(int dev = 1; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaMemcpy(hcounts[dev], dcounts[dev], count_counts * sizeof(uintll_t), cudaMemcpyDeviceToHost) );
    for(uint_t i = 0; i < count_counts; ++i)
      hcounts[0][i] += hcounts[dev][i];
    CHECK_ERROR( cudaFree(dcounts[dev]) );
    delete[] hcounts[dev];
  }

  // Process results into distance counts
  uint128_t counts[64] = {0};
  counts[0] = 1ull << n;
  counts[1] = with_1bit ? uint128_t(1ull << n) * (bitcount_message) : 0;
  counts[2] = 0;
  
  for(uint_t i = 4; i < count_counts; i += 2)
  {
    counts[i] = hcounts[0][i] + hcounts[0][i-1];
    counts[i-1] = 0;
  }
  
  if(with_1bit)
  {  
    // Calculate results for 1-bit sphere error detection/correction
    for (uint_t i = 3; i < count_counts; i += 2)
    {
      if(i + 1 < count_counts) {
        counts[i] = uint128_t(i + 1) * counts[i + 1] + uint128_t(bitcount_message - i + 1) * counts[i - 1];
      } else {
        counts[i] = uint128_t(bitcount_message - i + 1) * counts[i - 1];
      }
    }
  }

  for(uint_t i = 3; i < count_counts; ++i)
    counts[i] <<= n;

  results_cpu.stop(i_totaltime);

  if(verbose || file_output)
  {
    const char* prefix = file_output ? (nr_dev == 4 ? "hamming_4gpu" : "hamming_gpu") : nullptr;
    process_result_hamming(counts, stats, n, h, prefix);
  }
  
  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;

  CHECK_ERROR( cudaDeviceReset() );
}
