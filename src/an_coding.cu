
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
/*
template <uint_t BlockSize, typename T>
inline
__device__ void warpReduce(volatile T *sdata) {
  if (BlockSize >= 64) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
  if (BlockSize >= 32) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
  if (BlockSize >= 16) sdata[threadIdx.x] += sdata[threadIdx.x + 8];
  if (BlockSize >= 8) sdata[threadIdx.x] += sdata[threadIdx.x + 4];
  if (BlockSize >= 4) sdata[threadIdx.x] += sdata[threadIdx.x + 2];
  if (BlockSize >= 2) sdata[threadIdx.x] += sdata[threadIdx.x + 1];
}

template<uint_t BlockSize, typename T>
__device__ void dreduce(T* sdata) {
  if (BlockSize >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] += sdata[threadIdx.x + 256]; } __syncthreads(); }
  if (BlockSize >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] += sdata[threadIdx.x + 128]; } __syncthreads(); }
  if (BlockSize >= 128) { if (threadIdx.x < 64) { sdata[threadIdx.x] += sdata[threadIdx.x + 64]; } __syncthreads(); }
  if (threadIdx.x < 32)
    warpReduce<BlockSize>(sdata);
}
*/
// 4*0.875*32*64/8 = 896 GB/s (K80 Peak SMEM Bandwidth) (4x warp scheduler*gpufreq*banks*width)
// 4*0.784*32*64/8 = 800 GB/s (K20 **)

template<uint_t BlockSize, uint_t CountCounts, typename T, T Unroll>
__global__
void dancoding_shared(T n, T A, uintll_t* counts, T offset, T end, T Aend)
{
  constexpr uint_t SharedRowPitch = BlockSize;
  constexpr uint_t SharedCount = CountCounts * SharedRowPitch;
  __shared__ T counts_shared[SharedCount];

  T Aend_p = Aend-Unroll*A;
  for(uint_t i = threadIdx.x; i < SharedCount; i+=SharedRowPitch) {
    counts_shared[i] = 0;
  }
  __syncthreads();

  uint_t z[Unroll];
  T v, w, i;
  for (i = blockIdx.x * BlockSize + threadIdx.x + offset;
       i < end;
       i += BlockSize * gridDim.x)
  {
    w = A*i;
    for(v=w+A; v<Aend_p; v=v+Unroll*A)
    {
      #pragma unroll
      for(T u=0; u<Unroll; ++u) {
        z[u] = ANCoding::dbitcount( w^(v+u*A) )*SharedRowPitch + threadIdx.x;
        ++counts_shared[ z[u] ];
      }
    }
    for(; v<Aend; v+=A)
    {
      z[0] = ANCoding::dbitcount( w^v )*SharedRowPitch + threadIdx.x;
      ++counts_shared[ z[0] ];
    }
  }
  __syncthreads();
/*
  // reduction
  for(uint_t c=0; c<CountCounts; ++c)
    dreduce<BlockSize>(&counts_shared[c*SharedRowPitch]);

  if(threadIdx.x<CountCounts) {
    atomicAdd(&counts[threadIdx.x], counts_shared[threadIdx.x*SharedRowPitch]); // gives a bank conflict, but can be neglected
  }*/
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
  using value_type = typename std::conditional< (N<=24), uint_t, uintll_t >::type;
  static constexpr uint_t blocksize = std::is_same<value_type, uint_t>::value ? 128 : 64;
  static constexpr value_type unroll_factor = std::is_same<value_type, uint_t>::value ? 16 : 32;

  static constexpr cudaSharedMemConfig smem_config = std::is_same<value_type, uint_t>::value ? cudaSharedMemBankSizeFourByte : cudaSharedMemBankSizeEightByte;

  void operator()(uintll_t n, dim3 blocks, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end, uint_t ccmax){
    uintll_t Aend = A<<n;
    CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
    CHECK_ERROR( cudaDeviceSetSharedMemConfig(smem_config) );
    if(ccmax<16)
      dancoding_shared<blocksize, 16, value_type, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<24)
      dancoding_shared<blocksize, 24, value_type, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<32)
      dancoding_shared<blocksize, 32, value_type, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<48)
      dancoding_shared<blocksize, 48, value_type, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else if(ccmax<64)
      dancoding_shared<blocksize, 64, value_type, unroll_factor ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
    else
      throw std::runtime_error("Not supported");

/*      CHECK_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
      if(ccmax<16)
        dancoding<16, value_type ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
      else if(ccmax<24)
        dancoding<24, value_type ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
      else if(ccmax<32)
        dancoding<32, value_type ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
      else if(ccmax<48)
        dancoding<48, value_type ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
      else if(ccmax<64)
        dancoding<64, value_type ><<< blocks, blocksize >>>(n, A, counts, offset, end, Aend);
      else
      throw std::runtime_error("Not supported");*/
  }
};


void run_ancoding(uintll_t n, uintll_t A, int verbose, double* times, uintll_t* minb, uint128_t* mincb, int file_output, int nr_dev_max)
{
  uint_t h = ceil(log(A)/log(2.0));

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
  if(verbose>1){
    printf("Start AN-Coding Algorithm\n");
    printf("Found %d CUDA devices.\n", nr_dev);
  }
  std::stringstream ss;
  // skip init time
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaDeviceSynchronize() );
    cudaDeviceProp props;
    CHECK_ERROR( cudaGetDeviceProperties(&props, dev) );
    ss << "\"Device\", " << dev
       << ", \"MemClock [MHz]\", " << props.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << props.clockRate/1000
       << endl;

    CHECK_ERROR( cudaDeviceSynchronize() );
  }

  results_cpu.start(i_totaltime);

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

    if(verbose>1){
      CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
      printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d %d, offset %llu, end %llu, end %llu, load %Lg\n", dev, numSMs, blocks.x, blocks.y, offset, end, (threads.x-1+threads.x * ((blocks.x-1) * (blocks.x) + (blocks.x-1)) + offset), 1.0L*(end-offset)*(count_messages-(end+offset)/2.0L));
    }

    if(dev==0)
      results_gpu.start(i_runtime);
//    for(int run=0; run<10; ++run) // just for profiling
    ANCoding::bridge<Caller>(n, blocks, A, dcounts[dev], offset, end, count_counts);
    CHECK_LAST("Kernel failed.");

    if(dev==0)
      results_gpu.stop(i_runtime);

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
  //
  for(uint_t i=1; i<count_counts; ++i)
  {
    counts[i] = hcounts[0][i]<<1;
  }

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

  if(times!=NULL)
  {
    times[0] = stats.getAverage(0);
    times[1] = stats.getAverage(1);
  }

  if(verbose || file_output){
    if(nr_dev==4)
      process_result_ancoding(counts, stats, n, A, file_output?"ancoding_4gpu":nullptr,ss.str());
    else
      process_result_ancoding(counts, stats, n, A, file_output?"ancoding_gpu":nullptr,ss.str());
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;

  CHECK_ERROR( cudaDeviceReset() );

}
