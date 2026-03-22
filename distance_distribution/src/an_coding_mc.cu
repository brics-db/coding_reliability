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

/**
 * @brief CUDA kernel for AN-Coding distance distribution using Monte Carlo approximation.
 * 
 * Each thread samples random codewords to estimate the distance distribution.
 * 
 * @tparam ShardSize Size of the data shards for processing.
 * @tparam CountCounts The number of entries in the distance histogram.
 * @tparam T The numeric type for word values.
 * @tparam RandGenType The type of CURAND state used.
 * 
 * @param n Number of data bits.
 * @param A Multiplier for AN-coding.
 * @param counts Global distance histogram.
 * @param offset Starting codeword index.
 * @param end Ending codeword index.
 * @param state CURAND state array.
 * @param iterations Number of Monte Carlo iterations per thread.
 * @param p2n The value of 2^n for sampling.
 */
template<uintll_t ShardSize, uint_t CountCounts, typename T, typename RandGenType>
__global__
void dancoding_mc(T n, T A, uintll_t* counts, T offset, T end, RandGenType* state, uint_t iterations, double p2n)
{
  uint_t tid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
  T shardXid = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + offset;
  if(shardXid >= end)
    return;

  T counts_local[CountCounts] = { 0 };

  T w = A * shardXid * ShardSize;
  T wend = A * (shardXid + 1) * ShardSize;
  T v;
  T it;
  RandGenType local_state = state[tid];
  for(; w < wend; w += A)
  {
    for(it = 0; it < iterations; ++it)
    {
      v = static_cast<T>( p2n * curand_uniform_double(&local_state));
      v *= A;
      ++counts_local[ popc( w ^ v ) ];
    }
  }
  for(int c = 0; c < CountCounts; ++c)
    atomicAdd(counts + c, counts_local[c]);
  state[tid] = local_state;
}

/**
 * @brief Optimized CUDA kernel for AN-Coding Monte Carlo using shared memory histograms.
 * 
 * @tparam BlockSize Threads per block.
 * @tparam CountCounts Number of distance histogram bins.
 * @tparam T Word data type.
 * @tparam Unroll Loop unrolling factor.
 * @tparam RandGenType CURAND state type.
 */
template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll, typename RandGenType>
__global__
void dancoding_mc_shared(T n, T A, uintll_t* counts, T offset, T end, uint_t iterations, RandGenType* state, double p2n)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ uint_t counts_shared[SharedCount];

  // Initialize shared memory
  for(uint_t i = threadIdx.x; i < SharedCount; i += SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  T v, w, i;
  uint_t max = 0;
  RandGenType local_state = state[blockIdx.x * BlockSize + threadIdx.x];
  uint_t it;
  uint_t iterations_p = iterations <= Unroll ? 0 : iterations - Unroll;

  // Grid-stride loop
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end;
       i += BlockSize * gridDim.x)
  {
    w = A * i;
    for(it = 0; it < iterations_p; it += Unroll)
    {
      #pragma unroll
      for(uint_t u = 0; u < Unroll; ++u)
      {
        v = static_cast<T>( p2n * curand_uniform_double(&local_state));
        v *= A;
        uint_t z = ANCoding::dbitcount( w ^ v ) * SharedRowPitch + threadIdx.x;
        ++counts_shared[ z ];
        if(counts_shared[z] > max) max = counts_shared[z];
      }
      // Periodic flush to global to prevent shared memory overflow
      if(max > ((1u << 31) - Unroll) << 1) { 
        for(uint_t c = 0; c < CountCounts; ++c) {
          atomicAdd(&counts[c], counts_shared[c * SharedRowPitch + threadIdx.x]);
          max = 0;
          counts_shared[c * SharedRowPitch + threadIdx.x] = 0;
        }
      }
    }
    // Remainder of iterations
    for(; it < iterations; ++it)
    {
      v = static_cast<T>( p2n * curand_uniform_double(&local_state));
      v *= A;
      uint_t z = ANCoding::dbitcount( w ^ v ) * SharedRowPitch + threadIdx.x;
      ++counts_shared[ z ];
    }
  }
  __syncthreads();

  // Final aggregate to global memory
  if(threadIdx.x < CountCounts) {
    for(uint_t c = 0; c < BlockSize; ++c)
      atomicAdd(&counts[threadIdx.x], static_cast<uintll_t>(counts_shared[threadIdx.x * SharedRowPitch + c]));
  }

  state[blockIdx.x * BlockSize + threadIdx.x] = local_state;
}

/**
 * @brief Helper structure to launch the appropriate AN-Coding Monte Carlo kernel.
 */
template<uintll_t N>
struct Caller
{
  template<typename RandGenType>
  void operator()(uintll_t n, dim3 blocks, dim3 threads, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t ccmax, RandGenType* states, uintll_t iterations){
    static constexpr uint_t blocksize = 128;
    static constexpr uint_t unroll_factor = 16;
    double p2n = pow(2.0, (double)n);

    CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    CHECK_ERROR( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );

    if(ccmax <= 16)
      dancoding_mc_shared<blocksize, 16, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, iterations, states, p2n);
    else if(ccmax <= 24)
      dancoding_mc_shared<blocksize, 24, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, iterations, states, p2n);
    else if(ccmax <= 32)
      dancoding_mc_shared<blocksize, 32, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, iterations, states, p2n);
    else if(ccmax <= 48)
      dancoding_mc_shared<blocksize, 48, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, iterations, states, p2n);
    else if(ccmax <= 64)
      dancoding_mc_shared<blocksize, 64, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, iterations, states, p2n);
    else
      throw std::runtime_error("CountCounts not supported");
  }
};

/**
 * @brief Host function to run AN-Coding Monte Carlo approximation across multiple GPUs.
 * 
 * @param n Number of data bits.
 * @param iterations Number of samples to take.
 * @param A AN-coding multiplier.
 * @param verbose Verbosity level.
 * @param times Pointer for runtime statistics.
 * @param minb Pointer for minimum distance found.
 * @param mincb Pointer for count at minimum distance.
 * @param file_output Whether to output to a file.
 * @param nr_dev_max Maximum number of GPUs to use.
 * 
 * @return The maximum relative error achieved.
 */
double run_ancoding_mc(uintll_t n, uintll_t iterations, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb, int file_output, int nr_dev_max)
{
  if( ( A & (A - 1) ) == 0 ) // A is power of two
    A = 1;

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
    printf("Start AN-Coding Algorithm - Monte Carlo with %zu iterations\n", iterations);
    printf("Found %d CUDA devices.\n", nr_dev);
  }

  // Initialize devices
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  const uintll_t bitcount_A = ceil(log((double)A) / log(2.0));
  const uint_t count_counts = n + bitcount_A + 1;

  uintll_t** dcounts = new uintll_t*[nr_dev];
  uintll_t** hcounts = new uintll_t*[nr_dev];
  iterations = iterations > count_messages ? count_messages : iterations;

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    uintll_t offset, end;
    dim3 blocks;
    RandGen<RAND_GEN> gen;

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(&dcounts[dev], count_counts * sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts * sizeof(uintll_t)) );

    hcounts[dev] = new uintll_t[count_counts];
    memset(hcounts[dev], 0, count_counts * sizeof(uintll_t));

    offset = count_messages / nr_dev * dev;
    end = count_messages / nr_dev * (dev + 1);

    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );
    blocks.x = 128 * numSMs;

    if(verbose > 1) {
      cudaDeviceProp prop;
      CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
      printf("%d/%d threads on %s.\n", omp_get_thread_num() + 1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d, offset %llu, end %llu\n", dev, numSMs, blocks.x, offset, end);
    }

    // Initialize random number generator
    gen.init(blocks, threads, 16384 + 8137 * (end - offset) * dev, 1, dev);
    
    if(dev == 0) results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, threads, A, dcounts[dev], offset, end, count_counts, gen.devStates, iterations);
    CHECK_LAST("Kernel failed.");

    if(dev == 0) results_gpu.stop(i_runtime);

    gen.free();
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

  results_cpu.stop(i_totaltime);

  // Process Monte Carlo results: extrapolate counts based on sample density
  uint128_t counts[64] = {0};
  for(uint_t i = 0; i < count_counts; ++i)
  {
    counts[i] = static_cast<uint128_t>(static_cast<long double>(pow(2.0L, (long double)n) / iterations * hcounts[0][i]));
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;
  CHECK_ERROR( cudaDeviceReset() );

  // Estimate maximum relative error for the Monte Carlo sampling
  double max_abs_error = get_rel_error_AN(A, n, counts, 0);

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

  if(verbose || file_output)
  {
    const char* prefix = file_output ? (nr_dev == 4 ? "ancoding_mc_4gpu" : "ancoding_mc_gpu") : nullptr;
    process_result_ancoding_mc(counts, stats, n, A, iterations, prefix);
  }

  if(times != NULL)
  {
    times[0] = stats.getAverage(i_totaltime);
    times[1] = stats.getAverage(i_runtime);
  }

  return max_abs_error;
}
