
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
using namespace std;


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
 */
template<uintll_t N>
struct Caller
{
  void operator()(uintll_t n, dim3 blocks, dim3 threads, uintll_t A, uintll_t* counts, uintll_t offset, uintll_t end){
    uintll_t Aend = A<<n;
    if(Aend<(1ull<<32)){
      dancoding<ANCoding::traits::CountCounts<N>::value, uint_t >
        <<< blocks, threads >>>(n, A, counts, offset, end, Aend);
    }else{
      dancoding<ANCoding::traits::CountCounts<N>::value, uintll_t >
        <<< blocks, threads >>>(n, A, counts, offset, end, Aend);
    }
  }
};

void run_ancoding(uintll_t n, uintll_t A, int verbose, uintll_t* minb, uintll_t* mincb, int file_output, int nr_dev_max)
{
  uint_t h = ceil(log(A)/log(2.0));
  assert((n+h)<ANCoding::getCountCounts(n));

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

  // skip init time
  for(int dev=0; dev<nr_dev; ++dev)
  {
    CHECK_ERROR( cudaSetDevice(dev) );
    CHECK_ERROR( cudaThreadSetCacheConfig(cudaFuncCachePreferL1) );
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
    dim3 threads(128, 1, 1);
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
    blocks.x = 32*numSMs;

    if(verbose>1){
      CHECK_ERROR( cudaGetDeviceProperties(&prop, dev) );
      printf("%d/%d threads on %s.\n", omp_get_thread_num()+1, omp_get_num_threads(), prop.name);
      printf("Dev %d: SMs: %d Blocks: %d %d, offset %llu, end %llu, end %llu, load %Lg\n", dev, numSMs, blocks.x, blocks.y, offset, end, (threads.x-1+threads.x * ((blocks.x-1) * (blocks.x) + (blocks.x-1)) + offset), 1.0L*(end-offset)*(count_messages-(end+offset)/2.0L));
    }

    if(dev==0)
      results_gpu.start(i_runtime);
 
    ANCoding::bridge<Caller>(n, blocks, threads,A,dcounts[dev], offset, end);
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
    *minb=0xFFFF;;
    *mincb=0xFFFFFFFF;
    for(uint_t i=1; i<count_counts/2; ++i)
    {
      if(counts[i]!=0 && counts[i]<*mincb)
      {
        *minb=i;
        *mincb=counts[i];
      }
    }
  }

  if(verbose || file_output){
    if(nr_dev==4)
      process_result_ancoding(counts, stats, n, A, file_output?"ancoding_4gpu":nullptr);
    else
      process_result_ancoding(counts, stats, n, A, file_output?"ancoding_gpu":nullptr);
  }

  CHECK_ERROR( cudaFree(dcounts[0]) );
  delete[] hcounts[0];
  delete[] hcounts;
  delete[] dcounts;
  CHECK_ERROR( cudaDeviceReset() );

}
