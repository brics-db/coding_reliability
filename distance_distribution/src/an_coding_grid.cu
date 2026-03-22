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

/**
 * @brief CUDA kernel for AN-Coding distance distribution using a 1D grid approximation.
 * 
 * Instead of full combinatorial comparison or random sampling, this kernel samples
 * codewords at regular intervals (a grid) to approximate the distribution.
 */
template<uint_t CountCounts, typename T, typename TReal>
__global__
void dancoding_grid_1d(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize)
{
  T counts_local[CountCounts] = { 0 };
  T v, w, k;
  // Grid-stride loop
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < end;
       i += blockDim.x * gridDim.x)
  {
    w = A * i;
    for(v = 0, k = 0; v < Aend; ++k, v = A * static_cast<T>(k * stepsize))
    {
      ++counts_local[ ANCoding::dbitcount( w ^ v ) ];
    }
  }
  // Aggregate local histogram to global memory
  for(uint_t c = 1; c < CountCounts; ++c)
    atomicAdd(counts + c, counts_local[c]);
}

/**
 * @brief Optimized 1D grid kernel using shared memory histograms.
 */
template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll, typename TReal>
__global__
void dancoding_grid_1d_shared(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ uint_t counts_shared[SharedCount];

  T Aend_p = Aend - A * static_cast<T>(Unroll * stepsize);
  // Initialize shared memory
  for(uint_t i = threadIdx.x; i < SharedCount; i += SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  T v, w, i, k;
  uint_t max = 0;
  // Grid-stride loop
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end; 
       i += BlockSize * gridDim.x)
  {
    w = A * i;
    for(v = 0, k = 0; v < Aend_p; k += Unroll)
    {
      #pragma unroll (Unroll)
      for(T u = 0; u < Unroll; ++u, v = A * static_cast<T>((k + u) * stepsize))
      {
        uint_t z = ANCoding::dbitcount( w ^ v ) * SharedRowPitch + threadIdx.x;
        ++counts_shared[ z ];
        if(counts_shared[z] > max) max = counts_shared[z];
      }
      // Overflow prevention
      if(max > ((1u << 31) - Unroll) << 1) { 
        for(uint_t c = 0; c < CountCounts; ++c) {
          atomicAdd(&counts[c], counts_shared[c * SharedRowPitch + threadIdx.x]);
          max = 0;
          counts_shared[c * SharedRowPitch + threadIdx.x] = 0;
        }
      }
    }
    // Handle remaining steps
    for(; v < Aend; ++k, v = A * static_cast<T>(k * stepsize))
    {
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
}

/**
 * @brief CUDA kernel for AN-Coding distance distribution using a 2D grid approximation.
 * 
 * Samples both words in the comparison pair from a grid.
 */
template<uint_t CountCounts, typename T, typename TReal>
__global__
void dancoding_grid_2d(T n, T A, uintll_t* counts, T offset, T end, T Aend, TReal stepsize, TReal stepsize2)
{
  T counts_local[CountCounts] = { 0 };
  T v, w, k;
  // Grid-stride loop
  for (T i = blockIdx.x * blockDim.x + threadIdx.x + offset; 
       i < end; 
       i += blockDim.x * gridDim.x) 
  {
    w = A * static_cast<T>(i * stepsize2);
    for(v = 0, k = 0; v < Aend; ++k, v = A * static_cast<T>(k * stepsize))
    {
      ++counts_local[ ANCoding::dbitcount( w ^ v ) ];
    }
  }
  for(int c = 1; c < CountCounts; ++c)
    atomicAdd(counts + c, counts_local[c]);
}

/**
 * @brief Helper structure to launch the appropriate AN-Coding Grid kernel.
 */
template<uintll_t N>
struct Caller
{
  template<typename T>
  void operator()(uintll_t n, dim3 blocks, dim3 threads, int gdim, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t ccmax, T stepsize, T stepsize2, uint_t iterations){
    static constexpr uint_t blocksize = 128;
    static constexpr uint_t unroll_factor = 16;
    uintll_t Aend = A << n;

    if(gdim == 1) {
      assert(iterations >= 3);
      CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
      CHECK_ERROR( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) );

      if(iterations <= 16) {
        if(ccmax <= 16)
          dancoding_grid_1d_shared<blocksize, 16, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 24)
          dancoding_grid_1d_shared<blocksize, 24, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 32)
          dancoding_grid_1d_shared<blocksize, 32, uint_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 48)
          dancoding_grid_1d_shared<blocksize, 48, uintll_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 64)
          dancoding_grid_1d_shared<blocksize, 64, uintll_t, 3 ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else
          throw std::runtime_error("CountCounts not supported");
      } else {
        if(ccmax <= 16)
          dancoding_grid_1d_shared<blocksize, 16, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 24)
          dancoding_grid_1d_shared<blocksize, 24, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 32)
          dancoding_grid_1d_shared<blocksize, 32, uint_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 48)
          dancoding_grid_1d_shared<blocksize, 48, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else if(ccmax <= 64)
          dancoding_grid_1d_shared<blocksize, 64, uintll_t, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend, stepsize);
        else
          throw std::runtime_error("CountCounts not supported");
      }

    } else {
      if(Aend < (1ull << 32))
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uint_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
      else
        dancoding_grid_2d<ANCoding::traits::CountCounts<N>::value, uintll_t >
          <<< blocks, threads >>>(n, A, counts, offset, end, Aend, stepsize, stepsize2);
    }
  }
};

/**
 * @brief Host function to run AN-Coding Grid approximation across multiple GPUs.
 * 
 * @param gdim Grid dimensionality (1 or 2).
 * @param n Number of data bits.
 * @param iterations Number of grid points in the first dimension.
 * @param iterations2 Number of grid points in the second dimension (for 2D grid).
 * @param A AN-coding multiplier.
 * @param verbose Verbosity level.
 * @param times Output pointer for runtime statistics.
 * @param minb Output pointer for minimum distance.
 * @param mincb Output pointer for count at minimum distance.
 * @param file_output Whether to output results to a file.
 * @param nr_dev_max Maximum number of GPUs to use.
 * 
 * @return Estimated maximum relative error.
 */
double run_ancoding_grid(int gdim, uintll_t n, uintll_t iterations, uintll_t iterations2, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb, int file_output, int nr_dev_max)
{
  if( ( A & (A - 1) ) == 0 ) // A is power of two
    A = 1;
  assert(A < (1ull << n));

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
    printf("Start AN-Coding Algorithm - %dD Grid with %llu x %llu points\n", gdim, iterations, gdim == 2 ? iterations2 : 1);
    printf("Found %d CUDA devices.\n", nr_dev);
  }

  std::stringstream ss;
  // Initialize devices
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
    cudaDeviceProp props;
    CHECK_ERROR( cudaGetDeviceProperties(&props, dev) );
    ss << "\"Device\", " << dev
       << ", \"MemClock [MHz]\", " << props.memoryClockRate / 1000
       << ", \"GPUClock [MHz]\", " << props.clockRate / 1000
       << std::endl;

    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

  const uintll_t count_messages = (1ull << n);
  iterations = iterations > count_messages ? count_messages : iterations;
  iterations2 = gdim == 1 ? count_messages : iterations2 > count_messages ? count_messages : iterations2;

  const uintll_t bitcount_A = ceil(log((double)A) / log(2.0));
  const uint_t count_counts = n + bitcount_A + 1;

  uintll_t** dcounts = new uintll_t*[nr_dev];
  uintll_t** hcounts = new uintll_t*[nr_dev];

#pragma omp parallel for num_threads(nr_dev) schedule(static,1)
  for(int dev = 0; dev < nr_dev; ++dev)
  {
    dim3 threads(128, 1, 1);
    uintll_t offset, end;
    dim3 blocks;

    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaMalloc(&dcounts[dev], count_counts * sizeof(uintll_t)) );
    CHECK_ERROR( cudaMemset(dcounts[dev], 0, count_counts * sizeof(uintll_t)) );

    int numSMs;
    CHECK_ERROR( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev) );
    
    hcounts[dev] = new uintll_t[count_counts];

    offset = iterations2 / nr_dev * dev;
    end = iterations2 / nr_dev * (dev + 1);

    blocks.x = 128 * numSMs;

    if(verbose > 1) {
      cudaDeviceProp prop;
      CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
      printf("%d/%d threads on %s.\n", omp_get_thread_num() + 1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d, offset %llu, end %llu\n", dev, numSMs, blocks.x, offset, end);
    }
    
    if(dev == 0) results_gpu.start(i_runtime);

    ANCoding::bridge<Caller>(n, blocks, threads, gdim, A, dcounts[dev], offset, end, count_counts, 1.0 * count_messages / iterations, 1.0 * count_messages / iterations2, iterations);
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

  results_cpu.stop(i_totaltime);

  // Extrapolate counts based on grid density
  uint128_t counts[64] = {0};
  counts[0] = 1ull << n;
  long double factor = gdim == 1 ? pow(2.0L, (long double)n) / iterations : pow(4.0L, (long double)n) / (iterations * iterations2);
  for(uint_t i = 0; i < count_counts; ++i)
  {
    counts[i] = static_cast<uint128_t>(factor * hcounts[0][i]);
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;
  CHECK_ERROR( cudaDeviceReset() );

  // Relative error estimation
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
    char fname[256];
    sprintf(fname, "ancoding_grid%dd_%s", gdim, nr_dev == 4 ? "4gpu" : "gpu");
    process_result_ancoding_mc(counts, stats, n, A, iterations, file_output ? fname : nullptr, ss.str());
  }

  if(times != NULL)
  {
    times[0] = stats.getAverage(i_totaltime);
    times[1] = stats.getAverage(i_runtime);
  }

  return max_abs_error;
}
