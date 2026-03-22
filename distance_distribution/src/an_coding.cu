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

/**
 * @brief CUDA kernel for AN-Coding distance distribution using shared memory for efficiency.
 *
 * This kernel calculates the Hamming distance between pairs of AN-coded words.
 * It uses a shared memory histogram to minimize atomic collisions on global memory.
 * 
 * @tparam BlockSize The number of threads in each block.
 * @tparam CountCounts The number of entries in the distance histogram (n + h + 1).
 * @tparam T The numeric type for word values (uint_t or uintll_t).
 * @tparam Unroll Loop unrolling factor for performance optimization.
 * 
 * @param n Number of data bits.
 * @param A The AN-coding multiplier.
 * @param counts Global pointer to the distance histogram.
 * @param offset Starting index for this block's range of words.
 * @param end Ending index for the range of words.
 * @param Aend Maximum codeword value.
 */
template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll>
__global__
void dancoding_shared(T n, T A, uintll_t* counts, T offset, T end, T Aend)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ uint_t counts_shared[SharedCount];
  T Aend_p = Aend - Unroll * A;

  // Initialize shared memory histogram
  for(uint_t i = threadIdx.x; i < SharedCount; i += SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  T v, w, i;
  uint_t max = 0;
  // Grid-stride loop to handle work distribution across blocks
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end;
       i += BlockSize * gridDim.x)
  {
    w = A * i;
    // Inner loop: compare current word w with all subsequent words v
    for(v = w + A; v < Aend_p; v = v + Unroll * A)
    {
      #pragma unroll (Unroll)
      for(T u = 0; u < Unroll; ++u) {
        uint_t z = ANCoding::dbitcount( w ^ (v + u * A) ) * SharedRowPitch + threadIdx.x;
        ++counts_shared[ z ];
        if(counts_shared[z] > max) max = counts_shared[z];
      }
      // Periodically flush shared memory to global memory to prevent 32-bit overflow
      if(max > ((1u << 31) - Unroll) << 1) { 
        for(uint_t c = 0; c < CountCounts; ++c) {
          atomicAdd(&counts[c], counts_shared[c * SharedRowPitch + threadIdx.x]);
          max = 0;
          counts_shared[c * SharedRowPitch + threadIdx.x] = 0;
        }
      }
    }
    // Handle remaining words in the inner loop
    for(; v < Aend; v += A)
    {
      uint_t z = ANCoding::dbitcount( w ^ v ) * SharedRowPitch + threadIdx.x;
      ++counts_shared[ z ];
    }
  }
  __syncthreads();

  // Final flush of shared memory results to global memory
  if(threadIdx.x < CountCounts) {
    for(uint_t c = 0; c < BlockSize; ++c)
      atomicAdd(&counts[threadIdx.x], static_cast<uintll_t>(counts_shared[threadIdx.x * SharedRowPitch + c]));
  }
}

/**
 * @brief Simple CUDA kernel for AN-Coding distance distribution without shared memory.
 * 
 * Uses local registers for a small histogram before merging into global memory.
 * 
 * @tparam CountCounts The number of entries in the distance histogram.
 * @tparam T The numeric type for word values.
 */
template<uint_t CountCounts, typename T>
__global__
void dancoding(T n, T A, uintll_t* counts, T offset, T end, T Aend)
{
  T counts_local[CountCounts] = { 0 };
  T v, w;
  // Grid-stride loop
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < end;
       i += blockDim.x * gridDim.x)
  {
    w = A * i;
    for(v = w + A; v < Aend; v += A)
    {
      ++counts_local[ ANCoding::dbitcount( w ^ v ) ];
    }
  }
  // Aggregate local results into global memory
  for(uint_t c = 1; c < CountCounts; ++c)
    atomicAdd(counts + c, counts_local[c]);
}

/**
 * @brief Helper structure to launch the appropriate AN-Coding kernel based on data width N.
 */
template<uintll_t N>
struct Caller
{
  void operator()(uintll_t n, dim3 blocks, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t h) {
    static constexpr uint_t blocksize = 128;
    static constexpr uint_t unroll_factor = 16;
    uintll_t Aend = A << n;
    uint_t ccmax = n + h + 1;

    // Optimize shared memory and cache configuration for the kernel
    CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    CHECK_ERROR( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );

    // Select kernel based on the maximum number of histogram bins
    if(ccmax <= 16)
      dancoding_shared<blocksize, 16, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax <= 24)
      dancoding_shared<blocksize, 24, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax <= 32)
      dancoding_shared<blocksize, 32, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax <= 48)
      dancoding_shared<blocksize, 48, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax <= 64)
      dancoding_shared<blocksize, 64, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else
      throw std::runtime_error("CountCounts not supported");
  }
};

/**
 * @brief Host function to orchestrate the AN-Coding distance distribution calculation across multiple GPUs.
 * 
 * @param n Number of data bits.
 * @param A Multiplier for AN-coding.
 * @param verbose Verbosity level.
 * @param times Output pointer for runtime statistics.
 * @param minb Output pointer for the minimum distance found.
 * @param mincb Output pointer for the count at the minimum distance.
 * @param file_output Whether to output results to a file.
 * @param nr_dev_max Maximum number of GPUs to use (0 for all).
 */
void run_ancoding(uintll_t n, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb, int file_output, int nr_dev_max)
{
  if( ( A & (A - 1) ) == 0 ) // A is power of two
    A = 1;
  uint_t h = ceil(log(A) / log(2.0));

  int tmp_nr_dev;
  Statistics stats;
  TimeStatistics results_cpu (&stats, CPU_WALL_TIME);
  TimeStatistics results_gpu (&stats, GPU_TIME);
  int i_runtime = results_gpu.add("Kernel Runtime", "s");
  int i_totaltime = results_cpu.add("Total Runtime", "s");
  results_cpu.setFactorAll(0.001);
  results_gpu.setFactorAll(0.001);

  CHECK_ERROR( cudaGetDeviceCount(&tmp_nr_dev) );
  const int nr_dev = nr_dev_max == 0 ? tmp_nr_dev : min(nr_dev_max, tmp_nr_dev);
  
  if(verbose > 1) {
    printf("Start AN-Coding Algorithm\n");
    printf("Found %d CUDA devices.\n", nr_dev);
  }
  
  std::stringstream ss;
  // Initialize each device
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
    cudaDeviceProp props;
    CHECK_ERROR( cudaGetDeviceProperties(&props, dev) );
    ss << "\"Device\", " << dev
       << ", \"MemClock [MHz]\", " << props.memoryClockRate / 1000
       << ", \"GPUClock [MHz]\", " << props.clockRate / 1000
       << endl;

    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  const uint_t count_counts = n + h + 1;

  uintll_t** dcounts = new uintll_t*[nr_dev];
  uintll_t** hcounts = new uintll_t*[nr_dev];

  // Parallelize across GPUs using OpenMP
#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    dim3 blocks(1);
    uintll_t offset, end;
    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(&dcounts[dev], count_counts * sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts * sizeof(uintll_t)) );

    hcounts[dev] = new uintll_t[count_counts];
    if(nr_dev > 1) {
      // Non-linear load balancing to account for the quadratic complexity over the range of codewords
      double faca = 1.0 - sqrt( 1.0 - static_cast<double>(dev) / nr_dev );
      double facb = 1.0 - sqrt( 1.0 - static_cast<double>(dev + 1) / nr_dev );
      offset = ceil(count_messages * faca);
      end    = ceil(count_messages * facb);
    } else {
      offset = 0;
      end = count_messages;
    }
    
    // Grid-stride loop strategy
    blocks.x = 128 * numSMs;

    if(verbose > 1) {
      cudaDeviceProp prop;
      CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
      printf("%d/%d threads on %s.\n", omp_get_thread_num() + 1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d, offset %llu, end %llu\n", dev, numSMs, blocks.x, offset, end);
    }

    if(dev == 0) results_gpu.start(i_runtime);
    
    // Call the template kernel bridge
    ANCoding::bridge<Caller>(n, blocks, A, dcounts[dev], offset, end, h);
    CHECK_LAST("Kernel failed.");

    if(dev == 0) results_gpu.stop(i_runtime);
  }

  // Aggregate results from all GPUs back to the first device's host buffer
  CHECK_ERROR( cudaMemcpy(hcounts[0], dcounts[0], count_counts * sizeof(uintll_t), cudaMemcpyDeviceToHost) );
  for(int dev = 1; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaMemcpy(hcounts[dev], dcounts[dev], count_counts * sizeof(uintll_t), cudaMemcpyDeviceToHost) );
    for(uint_t i = 0; i < count_counts; ++i)
      hcounts[0][i] += hcounts[dev][i];
    CHECK_ERROR( cudaFree(dcounts[dev]) );
    delete[] hcounts[dev];
  }

  results_cpu.stop(i_totaltime);

  // Final count processing
  uint128_t counts[64] = {0};
  counts[0] = 1ull << n;
  for(uint_t i = 1; i < count_counts; ++i)
  {
    counts[i] = static_cast<uint128_t>(hcounts[0][i]) << 1;
  }

  // Record minimum distance results if requested
  if(minb != nullptr && mincb != nullptr)
  {
    *minb = 0xFFFF;
    *mincb = static_cast<uint128_t>(-1);
    for(uint_t i = 1; i < count_counts / 2; ++i)
    {
      if(counts[i] != 0 && counts[i] < *mincb)
      {
        *minb = i;
        *mincb = counts[i];
        break;
      }
    }
  }

  if(times != NULL)
  {
    times[0] = stats.getAverage(i_totaltime);
    times[1] = stats.getAverage(i_runtime);
  }

  if(verbose || file_output) {
    const char* prefix = file_output ? (nr_dev == 4 ? "ancoding_4gpu" : "ancoding_gpu") : nullptr;
    process_result_ancoding(counts, stats, n, A, prefix, ss.str());
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;

  CHECK_ERROR( cudaDeviceReset() );
}
